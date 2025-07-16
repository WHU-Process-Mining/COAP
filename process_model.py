from enum import Enum
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from utils import ACTIVITY_KEY



class DiscoveryAlgorithm(Enum):
    IND = 1
    ILP = 2


def construct_process_model(discovery_algorithm, completed_traces):
    log = EventLog()
    for trace in completed_traces:
        log.append(Trace({ACTIVITY_KEY: activity} for activity in trace))

    if discovery_algorithm == DiscoveryAlgorithm.IND:
        variant = inductive_miner.Variants.IMf
        process_tree = inductive_miner.apply(log, variant=variant)
        process_model = pt_converter.apply(process_tree)
    elif discovery_algorithm == DiscoveryAlgorithm.ILP:
        variant = ilp_miner.Variants.CLASSIC
        process_model = ilp_miner.apply(log, variant=variant, parameters={variant.value.Parameters.SHOW_PROGRESS_BAR: False})
    return process_model   

# 流程模型对轨迹约束，生成下一个可执行的活动集合
def generate_executable_activities(process_model, ongoing_trace):
    net, im, fm = process_model

    trace = Trace()
    for act in ongoing_trace:
        # PM4Py 默认用 "concept:name" 来存事件的活动名称
        trace.append({"concept:name": act})
    log = EventLog()
    log.append(trace)

    # 2. 用 token replay 对齐
    parameters = {token_replay.Variants.TOKEN_REPLAY.value.Parameters.DISABLE_VARIANTS: False}
    result = token_replay.apply(log, net, im, fm, parameters=parameters)[0]

    # result is a list of dicts, one per trace; we have only one trace
    firing_sequence = result["activated_transitions"]  
    final_marking    = result["reached_marking"]

    # 3. 基于 final_marking 拿下一步可执行 activities
    enabled = result["enabled_transitions_in_marking"]
    return {t.label for t in enabled if t.label is not None}
