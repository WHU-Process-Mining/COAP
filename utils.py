import csv
from datetime import timedelta
from os import path, makedirs
import os
import random

import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import torch
import pandas as pd
from pathlib import Path

CASE_ID_KEY = "case:concept:name"
ACTIVITY_KEY = "concept:name"
TIMESTAMP_KEY = "time:timestamp"
FINAL_ACTIVITY = "_END_"
SEED = 42


def generate_csv(log_name, case_id=CASE_ID_KEY, activity=ACTIVITY_KEY, timestamp=TIMESTAMP_KEY, add_final_activity=True):
    """
    将输入的XES文件转换为CSV格式，每个trace通过定义的最终事件进行扩展，根据设置的名称提取case_id,activity,timestamp并统一化，按时间顺序排序为事件流，将日志的case_id按出现顺序编码成从1开始的数字
    :param log_name: 包含事件日志的XES文件（可能已压缩）的名称
    :param case_id: 流程实例标识符属性（前缀为“case:”）
    :param activity: 标识已执行活动的属性
    :param timestamp: 指示事件执行时刻的属性
    :return:
    """
    csv_path = path.join('eventlog', 'CSV', log_name + '.csv')
    if not path.isfile(csv_path):
        if log_name == 'helpdesk2017' or log_name == 'SP2020':
            org_file = path.join('eventlog', 'ORI', log_name + '.csv')
        else:
            org_file = path.join('eventlog', 'ORI', log_name + '.xes')
        if Path(org_file).suffix == '.csv':
            print('Generating CSV file from CSV log...')
            if log_name =='helpdesk2017':
                dataset_column = ['Case ID', 'Activity','Complete Timestamp']
                dataframe = pd.read_csv(org_file)
            elif log_name == 'SP2020':
                dataset_column = ['CASE_ID', 'ACTIVITY', 'TIMESTAMP']
                dataframe = pd.read_csv(org_file, sep=';')

            default = ['case:concept:name', 'concept:name', 'time:timestamp']
            dataframe.rename(columns=dict(zip(dataset_column, default)), inplace=True)
            
            dataframe[timestamp] = pd.to_datetime(dataframe[timestamp], utc=True)
            add_rows = []
            for case, g in dataframe.groupby(case_id, sort=False):
                last_ts = g[timestamp].iloc[-1]
                add_rows.append({
                    activity: FINAL_ACTIVITY,
                    case_id: case,
                    timestamp: last_ts + timedelta(seconds=1)
                })
            dataframe = pd.concat([dataframe, pd.DataFrame(add_rows)], ignore_index=True)
        else:
            print('Generating CSV file from XES log...')
            log = xes_importer.apply(org_file, variant=xes_importer.Variants.LINE_BY_LINE)
            for trace in log:
                trace.append({activity: FINAL_ACTIVITY, timestamp: trace[-1][timestamp] + timedelta(seconds=1)})
            dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        dataframe = dataframe.filter(items=[activity, timestamp, case_id]).sort_values(timestamp, kind='mergesort')
        dataframe = dataframe.rename(columns={activity: ACTIVITY_KEY, case_id: CASE_ID_KEY})
        makedirs(path.dirname(csv_path), exist_ok=True)
        dataframe.to_csv(csv_path, index=False)


# fix random seed for reproducibility
def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    generate_csv("BPIC2020PrepaidTravelCost")
