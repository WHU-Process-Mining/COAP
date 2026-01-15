from itertools import product
from os import path
import pickle

import torch

from online_prediction import OnlinePredictor
from prediction_model import PredictionAlgorithm
from process_model import DiscoveryAlgorithm
from utils import SEED, fix_seed, generate_csv


event_logs = [
    "Receipt",
    # "helpdesk2017",
    # "BPIC2012",
    # "BPIC2018_P",
    # "BPIC2020_PrepaidTravelCost",
    # "BPIC2020_RequestForPayment",
    # "BPIC2020_InternationalDeclarations",
    # "BPIC2020_PermitLog", 
]

discovery_algorithms = [
    DiscoveryAlgorithm.IND,
    DiscoveryAlgorithm.ILP
]

prediction_algorithms = [
    PredictionAlgorithm.LSTM,
    PredictionAlgorithm.Transformer
]

dynamic_update_settings = [
    True, 
    # False
]

update_strategy_settings = [
    'finetune', # default
    # 'retrain'
]

apply_constraint_settings = [
    True, 
    # False
]

use_consistency_settings = [
    True,
    # False
]

use_only_conflict_data_settings=[
    True,
   #  False
]

confidence_threshold_settings = [
    # 0.4,
    # 0.45,
    0.5, # default
    # 0.55,
    # 0.6,
]

consistency_alpha_settings= [
    0,
    0.25,
    0.5,
    0.75,
    1.0,
]

win_min_settings = [
    # 50,
    # 100,
    # 200,
    300, # default
    # 400,
    # 500
]


drift_threshold_settings= [
    0, # ~p<0.5 的阈值
    0.674, # ~p<0.25 的阈值
    1.282, # ~p<0.1 的阈值
    1.645, # ~p<0.05 的阈值
    2.326, # ~p<0.01 的阈值
    2.576, # ~p<0.005 的阈值
    3.090, # ~p<0.001 的阈值
]



for event_log in event_logs:
    if not path.isfile(path.join("event log", "CSV", event_log + ".csv")):
        generate_csv(event_log)
    print(f"Processing event log: {event_log}")
    
    best_accuracy = 0.0
    for discovery_algorithm, prediction_algorithm, dynamic_update, update_strategy, apply_constraint, use_consistency, use_only_conflict_data, confidence_threshold, consistency_alpha, win_min, drift_threshold in product(
        discovery_algorithms, prediction_algorithms, dynamic_update_settings, update_strategy_settings, apply_constraint_settings, use_consistency_settings, use_only_conflict_data_settings, confidence_threshold_settings, consistency_alpha_settings, win_min_settings, drift_threshold_settings
    ):
        print(
            f"event_log:{event_log}, discovery_algorithm:{discovery_algorithm.name}, prediction_algorithm:{prediction_algorithm.name}, dynamic_update:{dynamic_update}, update_strategy:{update_strategy}, apply_constraint:{apply_constraint}, use_consistency:{use_consistency}, use_only_conflict_data:{use_only_conflict_data}, confidence_threshold:{confidence_threshold}, consistency_alpha:{consistency_alpha}, win_min:{win_min}, adwin_threshold:{drift_threshold}"
        )
        fix_seed(SEED)
        torch_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # torch_device = torch.device("cpu")
        print(f'torch_device: {torch_device}')
        online_predictor = OnlinePredictor(torch_device, event_log, discovery_algorithm, prediction_algorithm, dynamic_update, update_strategy, apply_constraint, use_consistency,use_only_conflict_data, confidence_threshold, consistency_alpha, win_min, drift_threshold)
        online_predictor.process_event_stream()
        # 保存结果
        results = {
                    'discovery_algorithm': discovery_algorithm.name,
                    'prediction_algorithm': prediction_algorithm.name,
                    'dynamic_update': dynamic_update,
                    'update_strategy': update_strategy,
                    'apply_constraint': apply_constraint,
                    'use_consistency': use_consistency,
                    'use_only_conflict_data': use_only_conflict_data,
                    'confidence_threshold': confidence_threshold,
                    'consistency_alpha': consistency_alpha,
                    'win_min': win_min,
                    'drift_threshold': drift_threshold,
                    'prediction_accuracy': online_predictor.prediction_accuracy,
                    'constraint_accuracy_list': online_predictor.constraint_accuracy_list,
                    'prediction_consistency': online_predictor.prediction_consistency,
                    "predict_list": online_predictor.predict,
                    "ground_truth_list": online_predictor.ground_truth,
                    'test_event_idxs': online_predictor.test_event_idxs,
                    'drift_moments': online_predictor.drift_moments
                }
        
        if use_consistency and apply_constraint:
            accuracy = sum(online_predictor.prediction_accuracy) / len(online_predictor.prediction_accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"New best accuracy: {best_accuracy}")
                with open(f"experiments/results_{event_log}.pickle", "wb") as file:
                    pickle.dump(results, file)
        else:
            if apply_constraint:
                with open(f"experiments/results_{event_log}_no_consistency.pickle", "wb") as file:
                    pickle.dump(results, file)
            else:
                with open(f"experiments/results_{event_log}_no_constraint.pickle", "wb") as file:
                    pickle.dump(results, file)