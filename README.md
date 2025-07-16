# COAP

## Installation

1. Create a python environment

    ```bash
    conda create -n coap python=3.12
    conda activate coap
    ```

2. Install pytorch

    Following the official website's guidance (<https://pytorch.org/get-started/locally/>), install the corresponding PyTorch version based on your CUDA version. For our experiments, we use torch 2.6.0+cu116. The installation command is as follows:

    ```bash
    pip install torch==2.6.0+cu116 torchvision==0.21.0+cu116 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu116
    ```

3. Install other related dependencies

    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

First, you need to specify the `data_path` and `dataset` in the `/eventlog` directory.  
- `/eventlog/ORI`: stores the original files  
- `/eventlog/csv`: stores the event stream (CSV) files

## Running

```bash
python main.py
```
