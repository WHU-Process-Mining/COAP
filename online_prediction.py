from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from os import path, makedirs
import time
import numpy as np
import pandas
from prediction_model import PredicitionModel
from process_model import construct_process_model, generate_executable_activities
from utils import CASE_ID_KEY, ACTIVITY_KEY, FINAL_ACTIVITY
from pm4py.streaming.importer.csv.importer import apply as csv_stream_importer
from sklearn.metrics import f1_score

class OnlinePredictor:
    def __init__(
        self,
        torch_device,
        log_name,
        discovery_algorithm,
        prediction_algorithm,
        dynamic_update=True,
        update_strategy="finetune",
        apply_constraint=True,
        use_consistency=True,
        use_only_conflict_data=False,
        confidence_threshold=0.5,
        consistency_alpha=0.5,
        adwin_min=200,
        adwin_threshold=0.05,
    ) -> None:
        self.log_name = log_name
        self.event_stream = csv_stream_importer(path.join('eventlog', 'CSV', self.log_name + '.csv'))
        self.event_num = len(pandas.read_csv(path.join("eventlog", "CSV", self.log_name + ".csv")))
        self.cut_val = (self.event_num // 100) * 10
        self.discovery_algorithm = discovery_algorithm
        self.prediction_algorithm = prediction_algorithm
        self.prediction_model = PredicitionModel(torch_device, prediction_algorithm)
        self.process_model = None
        self.update_model_flag = False
        self.processed_events = 0
        self.completed_traces = OrderedDict()
        self.ongoing_traces = OrderedDict()
        self.next_activity_prediction_probabilities = OrderedDict()
        self.next_executable_activities = OrderedDict()
        self.next_activity_prediction = OrderedDict()
        self.drift_moments = []
        self.predict = []
        self.ground_truth = []
        self.prediction_accuracy = []
        self.constraint_accuracy_list = []
        self.test_event_idxs = []  # 记录每次测试的event的id
        self.adwin_cur = 0
        self.adwin_min = adwin_min
        self.adwin_max = adwin_min * 10
        self.adwin_threshold = adwin_threshold
        self.prediction_consistency = []
        self.dynamic_update = dynamic_update
        self.update_strategy = update_strategy
        self.apply_constraint = apply_constraint
        self.use_consistency = use_consistency
        self.use_only_conflict_data = use_only_conflict_data
        self.confidence_threshold = confidence_threshold  # 模型约束置信度(default=0.5)，筛选出可执行活动或预测概率大于阈值的条目
        self.consistency_alpha = consistency_alpha  # 一致性计算系数，default=0.5
        self.start_time = datetime.now()
        self.total_time = None

    def save_results_to_csv(self):
        columns = [
            "start_time",
            "log_name",
            "processed_events",
            "discovery_algorithm",
            "prediction_algorithm",
            "dynamic_update",
            "update_strategy",
            "apply_constraint",
            "use_consistency",
            "use_only_conflict_data",
            "confidence_threshold",
            "consistency_alpha",
            "adwin_min",
            "adwin_threshold",
            "drift_moments",
            "drift_count",
            "average_accuracy",
            "average_consistency",
            "average_fscore",
            "total_time",
        ]
        # 获取每个实验的结果，并将其存储到 experiment_results 列表中
        result = {
            "start_time": self.start_time,
            "log_name": self.log_name,
            "processed_events": self.processed_events,
            "discovery_algorithm": self.discovery_algorithm.name,
            "prediction_algorithm": self.prediction_algorithm.name,
            "dynamic_update": self.dynamic_update,
            "update_strategy": self.update_strategy,
            "apply_constraint": self.apply_constraint,
            "use_consistency": self.use_consistency,
            "use_only_conflict_data":self.use_only_conflict_data,
            "confidence_threshold": self.confidence_threshold,
            "consistency_alpha": self.consistency_alpha,
            "adwin_min": self.adwin_min,
            "adwin_threshold": self.adwin_threshold,
            "drift_moments": self.drift_moments,
            "drift_count": len(self.drift_moments),
            "average_accuracy": sum(self.prediction_accuracy) / len(self.prediction_accuracy),
            "average_consistency": sum(self.prediction_consistency) / len(self.prediction_consistency),
            "average_fscore": (round(f1_score(self.ground_truth, self.predict, average='macro'), 5)),
            "total_time": self.total_time,
        }

        filename = path.join("results", "result_" + self.log_name + ".csv")
        makedirs(path.dirname(filename), exist_ok=True)
        # 如果文件不存在，则创建并写入固定列结构
        if not path.isfile(filename):
            df = pandas.DataFrame(columns=columns)
            df.to_csv(filename, index=False)

        # 将新的实验结果追加到 CSV 文件中（作为一行）
        new_result_df = pandas.DataFrame([result], columns=columns)

        # 以附加模式打开文件（append mode），避免覆盖
        new_result_df.to_csv(filename, mode="a", header=False, index=False)
        print(f"Experiment results saved to {filename}")

    # TODO:融合模型约束的活动预测
    def predict_next_activity(self, case_id, ongoing_trace):
        next_activity_probabilities = self.prediction_model.predict(ongoing_trace)
        executable_activities = generate_executable_activities(self.process_model, ongoing_trace)

        # 对概率排序，取top-k
        k = 5
        top_k_activities = sorted(next_activity_probabilities.items(), key=lambda item: item[1], reverse=True)[:k]

        self.next_activity_prediction_probabilities[case_id] = top_k_activities
        self.next_executable_activities[case_id] = executable_activities

        # 预测模型给出的概率最高的活动
        max_activity = top_k_activities[0][0]

        # 筛选出可执行活动或预测概率大于阈值的条目
        filtered_probabilities = {k: v for k, v in next_activity_probabilities.items() if k in executable_activities or v > self.confidence_threshold}
        # 找到概率最高的活动
        max_constrainted_activity = (
            max(filtered_probabilities, key=filtered_probabilities.get) if len(filtered_probabilities) > 0 else top_k_activities[0][0]
        )

        if self.apply_constraint:
            return max_constrainted_activity
        else:
            return max_activity

    # TODO:计算流程模型的可执行活动集、预测模型的活动概率、真实活动之间的一致性
    def compute_prediction_consistency(self, case_id, activity):
        prediciton_accuracy = 1 if activity == self.next_activity_prediction[case_id] else 0
        constraint_accuracy = 1 if activity in self.next_executable_activities[case_id] else 0
        self.constraint_accuracy_list.append(constraint_accuracy)
        consistency = self.consistency_alpha * constraint_accuracy + (1 - self.consistency_alpha) * prediciton_accuracy
        self.prediction_consistency.append(consistency)

    # TODO:约束冲突检测概念漂移
    def detect_drift(self):
        if self.use_consistency:
            adwin = self.prediction_consistency
        else:
            adwin = self.prediction_accuracy  # DARWIN 只使用准确率作为漂移检测的指标
        cur = len(adwin)
        # if self.adwin_cur < cur - self.adwin_max: # 窗口太大会检测失效
        #     self.adwin_cur = cur - self.adwin_max
        old_start = self.adwin_cur
        while (
            cur - self.adwin_cur > self.adwin_min
            and abs(np.mean(adwin[(cur + self.adwin_cur) // 2 : cur]) - np.mean(adwin[self.adwin_cur : (cur + self.adwin_cur) // 2]))
            > self.adwin_threshold
        ):
            self.adwin_cur = (cur + self.adwin_cur) // 2
        if self.adwin_cur == old_start:
            return False
        else:
            return True

    def process_event_stream(self):
        start_time = time.process_time()
        print("------------Initialization------------")
        for event in self.event_stream:
            self.processed_events += 1
            case_id = event[CASE_ID_KEY]
            activity = event[ACTIVITY_KEY]
            # print(type(case_id),type(activity),type(event_time)) # 均为str类型
            if case_id not in self.ongoing_traces:
                self.ongoing_traces[case_id] = [[activity], self.processed_events]  # 记录轨迹最后一个事件的发生位置
            else:
                self.ongoing_traces[case_id][0].append(activity)
                self.ongoing_traces[case_id][1] = self.processed_events  # 记录轨迹最后一个事件的发生位置
                if activity == FINAL_ACTIVITY:
                    self.completed_traces[case_id] = self.ongoing_traces.pop(case_id)
            if self.processed_events == self.cut_val:
                training_set = deepcopy([row[0] for row in list(self.completed_traces.values())])
                self.process_model = construct_process_model(self.discovery_algorithm, training_set)
                training_set.extend(deepcopy([row[0] for row in list(self.ongoing_traces.values())]))
                self.prediction_model.retrain(training_set)
                self.completed_traces = OrderedDict()
                for case_id, (ongoing_trace, last_event) in self.ongoing_traces.items():
                    self.next_activity_prediction[case_id] = self.predict_next_activity(case_id, ongoing_trace)
                break

        print("------------Online ActivityPrediction------------")
        for event in self.event_stream:
            self.processed_events += 1
            case_id = event[CASE_ID_KEY]
            activity = event[ACTIVITY_KEY]

            if case_id not in self.ongoing_traces:
                self.ongoing_traces[case_id] = [[activity], self.processed_events]  # 记录轨迹最后一个事件的发生位置
                self.next_activity_prediction[case_id] = self.predict_next_activity(case_id, self.ongoing_traces[case_id][0])
            else:
                self.predict.append(self.next_activity_prediction[case_id])
                self.ground_truth.append(activity)
                self.prediction_accuracy.append(1 if activity == self.next_activity_prediction[case_id] else 0)
                self.compute_prediction_consistency(case_id, activity)
                self.test_event_idxs.append(self.processed_events)

                self.ongoing_traces[case_id][0].append(activity)
                if activity == FINAL_ACTIVITY:
                    self.completed_traces[case_id] = self.ongoing_traces.pop(case_id)
                    self.next_activity_prediction.pop(case_id)
                else:
                    self.next_activity_prediction[case_id] = self.predict_next_activity(case_id, self.ongoing_traces[case_id][0])

                if self.dynamic_update and self.detect_drift():
                    print("------------Updating Models------------")
                    self.drift_moments.append(self.processed_events)

                    if self.use_only_conflict_data:
                        training_set = deepcopy(
                            [row[0] for row in list(self.completed_traces.values()) if row[1] >= self.test_event_idxs[self.adwin_cur]]
                        )  # 只使用漂移点后的数据来更新模型
                    else:
                        training_set = deepcopy([row[0] for row in list(self.completed_traces.values())])
                    if len(training_set) > 0:
                        self.process_model = construct_process_model(self.discovery_algorithm, training_set)
                    training_set.extend(deepcopy([row[0] for row in list(self.ongoing_traces.values())]))
                    if self.update_strategy == "finetune":
                        self.prediction_model.update(training_set)
                    else:
                        self.prediction_model.retrain(training_set)
                    self.completed_traces = OrderedDict()
                    for case_id, (ongoing_trace, last_event) in self.ongoing_traces.items():
                        self.next_activity_prediction[case_id] = self.predict_next_activity(case_id, ongoing_trace)
    
        self.total_time = time.process_time() - start_time

        print("------------Saving Results------------")
        print("drift: ", len(self.drift_moments), self.drift_moments)
        print("average accuracy: ", sum(self.prediction_accuracy) / len(self.prediction_accuracy))
        print("average fscore: ", f1_score(self.ground_truth, self.predict, average='macro'))
        print("average consistency: ", sum(self.prediction_consistency) / len(self.prediction_consistency))
        print("total time: ", self.total_time)
        self.save_results_to_csv()
