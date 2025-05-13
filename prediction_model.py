from enum import Enum
from utils import FINAL_ACTIVITY
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

UNKNOWM_ACTIVITY = "_UNK_"
PADDING_ACTIVITY = "_PAD_"
PADDING_VALUE = 0  # 定义专用的填充值


class PredictionAlgorithm(Enum):
    LSTM = 1
    Transformer = 2


# 活动编码器
class ActivityEncoder:
    def __init__(self):
        self.activity_to_idx = {PADDING_ACTIVITY: 0, FINAL_ACTIVITY: 1, UNKNOWM_ACTIVITY: 2}
        self.idx_to_activity = {0: PADDING_ACTIVITY, 1: FINAL_ACTIVITY, 2: UNKNOWM_ACTIVITY}
        self.next_idx = 3

    def fit(self, traces):
        for trace in traces:
            for activity in trace:
                if activity not in self.activity_to_idx:
                    self.activity_to_idx[activity] = self.next_idx
                    self.idx_to_activity[self.next_idx] = activity
                    self.next_idx += 1

    def encode(self, traces):
        encoded_traces = []
        for trace in traces:
            encoded_traces.append([self.activity_to_idx.get(activity, self.activity_to_idx[UNKNOWM_ACTIVITY]) for activity in trace])
        return encoded_traces

    def decode(self, encoded_trace):
        return [self.idx_to_activity[idx] for idx in encoded_trace]


def preprocess_traces(traces, encoder):
    # TODO: 看手动填充到sequence_length的效果？
    X, y, lengths = [], [], []
    encoded_traces = encoder.encode(traces)

    for trace in encoded_traces:
        for i in range(len(trace) - 1):  # 不包含最后一个活动
            # padded_input = [PADDING_VALUE] * (sequence_length - (i + 1)) + trace[: (i + 1)]
            # X.append(padded_input[-sequence_length:])  # 截断到 sequence_length
            X.append(torch.tensor(trace[: (i + 1)], dtype=torch.long))
            y.append(trace[i + 1])
            lengths.append(i + 1)  # 记录有效长度
            # lengths.append(min(i + 1, sequence_length))  # 记录有效长度
    X = pad_sequence(X, batch_first=True, padding_value=PADDING_VALUE)  # 填充到最大长度，与后面的pack_padded_sequence配合使用
    return X, torch.tensor(y, dtype=torch.long), torch.tensor(lengths)


class LSTMActivityPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMActivityPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_VALUE)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, lengths):
        # 1. 嵌入层
        embedded = self.embedding(x)
        # 2. 压缩变长序列，以便忽略填充部分，仅计算实际的有效序列部分。
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        # 3. LSTM 前向传播
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # 4. 解压序列
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 5. 提取最后一个隐藏状态
        last_hidden = output[torch.arange(len(lengths)), lengths - 1]
        # 6. 全连接层输出
        return self.fc(last_hidden)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class TransformerActivityPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, dropout, max_seq_len=128):
        super(TransformerActivityPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_VALUE)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, lengths):
        # 1. 嵌入层
        embedded = self.embedding(x)
        # 2. 加入位置编码
        embedded = self.positional_encoding(embedded)
        # 3. 生成 mask: 对于每个 batch 中的序列，标记无效位置
        batch_size, seq_len = x.size(0), x.size(1)
        mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        mask = mask >= lengths.unsqueeze(1)  # True 表示需要忽略的位置
        # 4. Transformer Encoder 前向传播
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        # 5. 提取最后一个有效位置的表示
        last_hidden = encoded[torch.arange(len(lengths)), lengths - 1]
        # 6. 全连接层输出
        return self.fc(last_hidden)  # TODO:transformer预测是只用最后一个位置的还是所有的？


lstm_params = {"embedding_dim": 8, "hidden_dim": 32, "num_layers": 1, "dropout": 0, "epoch_num": 100, "batch_size": 64, "learning_rate": 0.01}
transformer_params = {
    "embedding_dim": 64,
    "hidden_dim": 512,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "epoch_num": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
}


class PredicitionModel:
    def __init__(self, torch_device, prediction_algorithm):
        self.device = torch_device
        self.prediction_algorithm = prediction_algorithm
        if self.prediction_algorithm == PredictionAlgorithm.LSTM:
            self.embedding_dim = lstm_params["embedding_dim"]
            self.hidden_dim = lstm_params["hidden_dim"]
            self.num_layers = lstm_params["num_layers"]
            self.dropout = lstm_params["dropout"]
            self.epoch_num = lstm_params["epoch_num"]
            self.batch_size = lstm_params["batch_size"]
            self.learning_rate = lstm_params["learning_rate"]
        elif self.prediction_algorithm == PredictionAlgorithm.Transformer:
            self.embedding_dim = transformer_params["embedding_dim"]
            self.hidden_dim = transformer_params["hidden_dim"]
            self.num_heads = transformer_params["num_heads"]
            self.num_layers = transformer_params["num_layers"]
            self.dropout = transformer_params["dropout"]
            self.epoch_num = transformer_params["epoch_num"]
            self.batch_size = transformer_params["batch_size"]
            self.learning_rate = transformer_params["learning_rate"]

        self.activity_encoder = ActivityEncoder()
        self.vocab_size = None
        self.model = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)
        self.optimizer = None

    def retrain(self, traces):
        # 活动编码
        self.activity_encoder = ActivityEncoder()
        self.activity_encoder.fit(traces)
        # 轨迹处理
        X, y, lengths = preprocess_traces(traces, self.activity_encoder)
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X, y, lengths)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # 模型和优化器
        self.vocab_size = len(self.activity_encoder.activity_to_idx)
        if self.prediction_algorithm == PredictionAlgorithm.LSTM:
            self.model = LSTMActivityPredictor(self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers, self.dropout)
        elif self.prediction_algorithm == PredictionAlgorithm.Transformer:
            self.model = TransformerActivityPredictor(
                self.vocab_size, self.embedding_dim, self.num_heads, self.num_layers, self.hidden_dim, self.dropout
            )
        self.model = self.model.to(self.device)  # 移动模型到指定设备
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # 模型训练
        self.train_model(dataloader, self.epoch_num)

    def update(self, new_traces):
        # 为可能的新活动更改活动的嵌入和模型的输出维度
        old_activity_num = len(self.activity_encoder.activity_to_idx)
        self.activity_encoder.fit(new_traces)
        new_activity_num = len(self.activity_encoder.activity_to_idx)
        if new_activity_num > old_activity_num:
            print("update activity num: ", old_activity_num, " -> ", new_activity_num)
            self.update_embedding_layer(new_activity_num)
            self.update_output_layer(new_activity_num)
        # 轨迹处理
        X, y, lengths = preprocess_traces(new_traces, self.activity_encoder)
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X, y, lengths)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # 设置优化器（仅更新 Embedding 或全模型）
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # TODO:这里用adam比adamw效果好是为什么
        # 微调
        self.train_model(dataloader, self.epoch_num // 5)

    def predict(self, ongoing_trace):
        self.model.eval()
        with torch.no_grad():
            encoded_trace = self.activity_encoder.encode([ongoing_trace])[0]
            input_seq = torch.tensor([encoded_trace], dtype=torch.long, device=self.device)
            input_length = torch.tensor([len(encoded_trace)])
            output = self.model(input_seq, input_length)
            probabilities = F.softmax(output, dim=-1).squeeze(0)  # 转为概率分布
            # 将索引映射为活动，并附带概率
            activity_probabilities = {self.activity_encoder.idx_to_activity[idx]: prob.item() for idx, prob in enumerate(probabilities)}
            return activity_probabilities

    def train_model(self, dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, y_batch, lengths_batch in dataloader:
                # 将数据移动到设备, 'lengths' argument should be a 1D CPU int64 tensor
                X_batch, y_batch, lengths_batch = X_batch.to(self.device), y_batch.to(self.device), lengths_batch
                predictions = self.model(X_batch, lengths_batch)
                loss = self.criterion(predictions, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    def update_embedding_layer(self, new_vocab_size):
        if new_vocab_size > self.model.embedding.num_embeddings:
            with torch.no_grad():
                # 随机初始化新的权重
                new_weights = torch.randn(
                    new_vocab_size - self.model.embedding.num_embeddings, self.model.embedding.embedding_dim, device=self.device
                )
                # 扩展当前的嵌入层权重
                updated_weights = torch.cat([self.model.embedding.weight.data, new_weights], dim=0)
                self.model.embedding = nn.Embedding.from_pretrained(updated_weights, freeze=False).to(self.device)

    def update_output_layer(self, new_vocab_size):
        old_vocab_size = self.model.fc.out_features
        if new_vocab_size > old_vocab_size:
            with torch.no_grad():
                # 保存旧的权重和偏置
                old_weights = self.model.fc.weight.data
                old_bias = self.model.fc.bias.data

                # 初始化新的权重和偏置
                new_weights = torch.randn(new_vocab_size - old_vocab_size, self.model.fc.in_features, device=self.device)
                new_bias = torch.randn(new_vocab_size - old_vocab_size, device=self.device)

                # 扩展并更新
                updated_weights = torch.cat([old_weights, new_weights], dim=0)
                updated_bias = torch.cat([old_bias, new_bias], dim=0)

                # 创建新的全连接层
                self.model.fc = nn.Linear(self.model.fc.in_features, new_vocab_size, device=self.device)
                self.model.fc.weight = nn.Parameter(updated_weights)
                self.model.fc.bias = nn.Parameter(updated_bias)
