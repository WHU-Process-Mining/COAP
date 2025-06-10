from collections import defaultdict
from enum import Enum
import pm4py
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.petri_net.semantics import weak_execute, enabled_transitions
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


# 模型约束的形式是{a_i:(pre_required(a_i),next_executable(a_i)) | a_i∈A}
def construct_model_constraints(process_model):
    model_constraints = defaultdict(lambda: [set(), set()])
    log = pm4py.play_out(*process_model, parameters={pm4py.algo.simulation.playout.petri_net.variants.basic_playout.Parameters.NO_TRACES: 100})

    for trace in log:
        pre_activity = None  # TODO：考虑完整prefix会变差是为什么
        for event in trace:
            activity = event[ACTIVITY_KEY]
            if pre_activity:
                    model_constraints[pre_activity][1].add(activity)
            pre_activity = activity
    return model_constraints

    # pre_required构建耗时长，还没啥效果

    # dependency_graph = networkx.DiGraph()
    # # 遍历约束，构建图
    # for pre_activity, (_, post_activities) in model_constraints.items():
    #     for activity in post_activities:
    #         dependency_graph.add_edge(pre_activity, activity)
    # try:
    #     topological_order = list(networkx.topological_sort(dependency_graph)) # 使用networkx的拓扑排序
    #     # 创建一个字典，存储每个活动的层级
    #     activity_levels = {node: None for node in dependency_graph.nodes}
    #     # 为每个活动分配层级
    #     for activity in topological_order:
    #         # 获取当前活动的前置活动
    #         predecessors = list(dependency_graph.predecessors(activity))
    #         if predecessors:
    #             # 如果有前置活动，则当前活动的层级为前置活动的最大层级 + 1
    #             activity_levels[activity] = max(activity_levels[predecessor] for predecessor in predecessors) + 1
    #         else:
    #             # 如果没有前置活动，当前活动可以处于第0层级
    #             activity_levels[activity] = 0
    # except networkx.NetworkXUnfeasible:
    #     activity_levels = None

    # if activity_levels:
    #     # 根据层级分组活动
    #     level_groups = defaultdict(list)
    #     for activity, level in activity_levels.items():
    #         level_groups[level].append(activity)

    #     # 活动的前序活动为level-1
    #     for activity in model_constraints:
    #         if activity_levels[activity]>0:
    #             model_constraints[activity][0].update(level_groups[activity_levels[activity]-1])

    # return model_constraints


# 流程模型对轨迹约束，生成下一个可执行的活动集合
def generate_executable_activities(process_model, process_model_constraints, ongoing_trace):
    return process_model_constraints[ongoing_trace[-1]][1] if ongoing_trace[-1] in process_model_constraints else set()

    # last_activity = ongoing_trace[-1]
    # if last_activity not in process_model_constraints:
    #     return set()

    # # 获取下一步可执行活动
    # next_executable_activities = {
    #     activity for activity in process_model_constraints[last_activity][1]
    #     if process_model_constraints[activity][0].issubset(ongoing_trace)
    # }
    # return next_executable_activities

    # net, initial_marking, final_marking = process_model

    # executable_activities = set()
    # # 提取ongoing_trace最后一个活动对应的transition的下一个可达的transition对应的活动
    # transitions = [t for t in net.transitions if t.label == ongoing_trace[-1]]
    # visited_t = set()
    # while len(transitions) > 0:
    #     t = transitions.pop(0)
    #     next_transitions = [p_t_arc.target for t_p_arc in t.out_arcs for p_t_arc in t_p_arc.target.out_arcs]
    #     for next_t in next_transitions:
    #         if next_t.label != None:
    #             executable_activities.add(next_t.label)
    #         elif next_t not in visited_t:
    #             visited_t.add(next_t)
    #             transitions.append(next_t)

    # #TODO：上述只考虑ongoing_trace最后一个活动，考虑逐步执行考虑整个轨迹（但是遇到不可见活动无法往下走）
    # net, initial_marking, final_marking = process_model
    # # 使用 Petri 网的初始标记作为当前标记
    # current_marking = initial_marking.copy()
    # # 遍历轨迹并更新标记
    # for activity in ongoing_trace:
    #     for transition in net.transitions:
    #         if transition.label == activity:
    #             current_marking=weak_execute(transition,current_marking)
    #             break
    # # 获取当前标记下可执行的活动（转移）
    # enabled_t = enabled_transitions(net, current_marking)
    # # 输出可执行的活动
    # executable_activities = set([transition.label for transition in enabled_t])

    # return executable_activities
