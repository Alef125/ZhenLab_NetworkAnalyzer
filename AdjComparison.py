# import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json


CONNECTION_THRESHOLD = 2


def not_nan(x):  # Only for int variables
    return isinstance(x, int)


def decide_flag(flags_list: list) -> int:
    # if len(flags_list) < 3:
    #     return 0
    all_false = True
    all_true = True
    for _flag in flags_list:
        if _flag:  # True
            all_false = False
        else:  # False
            all_true = False
    if all_true:
        return 1
    if all_false:
        return -1
    return 0


def extract_connections(connected_component: list, list_of_edges: list) -> list:
    including_edges = []
    for _edge in list_of_edges:
        pre_neuron, post_neuron = _edge[0], _edge[1]
        if pre_neuron in connected_component and post_neuron in connected_component:  # and = or, if components are true
            including_edges.append(_edge)
    return including_edges


def draw_graph(list_of_edges: list, saving_filepath: str) -> None:
    g = nx.DiGraph()
    g.add_edges_from(list_of_edges)
    plt.figure(figsize=(9, 9))
    nx.draw_networkx(g)
    # plt.show()
    plt.savefig(saving_filepath)


def adjacency_list_to_connected_components(adjacency_list: list) -> list:
    connected_components = []
    for added_connection in adjacency_list:
        pre_neuron, post_neuron = added_connection[0], added_connection[1]
        cmp_number = 0
        pre_group = None
        post_group = None
        break_flag = False
        for connected_component in connected_components:
            if pre_neuron in connected_component:
                if post_neuron in connected_component:
                    break_flag = True
                    break  # These two neurons are already in a same component
                pre_group = cmp_number
            if post_neuron in connected_component:
                if pre_neuron in connected_component:
                    break_flag = True
                    break  # These two neurons are already in a same component
                post_group = cmp_number
            cmp_number += 1
        if break_flag:
            continue  # Skipping the following section
        # ############### Add, append, or merge the new connection ###############
        if not_nan(pre_group):
            if not_nan(post_group):  # Merge two components
                connected_components[pre_group] += connected_components[post_group]
                del connected_components[post_group]
            else:  # pre_neuron in some component, post_neuron in no component
                connected_components[pre_group].append(post_neuron)
        else:
            if not_nan(post_group):  # pre_neuron in no component, post_neuron in some component
                connected_components[post_group].append(pre_neuron)
            else:  # Neither pre_neuron and post_neuron in any component
                if pre_neuron != post_neuron:
                    connected_components.append([pre_neuron, post_neuron])
                else:
                    connected_components.append([pre_neuron])  # Exception! SDQ self connection (actually, R <-> L)
        # ################################################################
    return connected_components


# class ConnectedComponent:
#     def __init__(self, init_connection: tuple):
#         self.neurons = [init_connection[0], init_connection[1]]
#         self.connections = [init_connection]
#
#     def does_contain_neuron(self, neuron_name: str):
#         return neuron_name in self.neurons
#
#     def does_share_neuron(self, connection_pair: tuple):
#         return connection_pair[0] in self.neurons or connection_pair[1] in self.neurons
#
#     def add_connection(self, connection_pair: tuple):
#         # Note: Check does_share_neuron externally
#         if not connection_pair[0] in self.neurons:
#             self.neurons.append(connection_pair[0])
#         if not connection_pair[1] in self.neurons:
#             self.neurons.append(connection_pair[1])
#         self.connections.append(connection_pair)
#
#     def merge_components(self,
#                          component_to_merge: ConnectedComponent,
#                          sharing_connection: tuple):
#         self.neurons += component_to_merge.neurons
#         self.connections += component_to_merge.connections
#         self.connections.append(sharing_connection)  # Neurons should be in either self or component_to_merge


class AdjacencyComparator:
    def __init__(self, interest_matrices_filepath: list, other_matrices_filepath: list):
        # self.adjacency_matrices = []
        # self.matrix_of_interest = None
        self.other_matrices = []
        self.interest_matrices = []
        self.load_adjacency_matrices(interest_matrices_filepath, other_matrices_filepath)
        self.all_neurons = self.interest_matrices[0].columns.tolist()  # Should be same in all
        # ###########################################################
        self.added_connections = []
        self.removed_connections = []
        # #############################################################
        self.added_connected_components = []
        self.removed_connected_components = []
        # #############################################################
        self.added_components_edges = []
        self.removed_components_edges = []

    def load_adjacency_matrices(self,
                                interest_matrices_filepath: list,
                                other_matrices_filepath: list) -> None:
        for matrix_filepath in interest_matrices_filepath:
            adjacency_matrix = pd.read_csv(matrix_filepath, index_col=[0])
            self.interest_matrices.append(adjacency_matrix)
        for matrix_filepath in other_matrices_filepath:
            adjacency_matrix = pd.read_csv(matrix_filepath, index_col=[0])
            self.other_matrices.append(adjacency_matrix)
            # if matrix_filepath == matrix_of_interest_filepath:
            #     self.matrix_of_interest = adjacency_matrix
            # else:
            #     self.adjacency_matrices.append(adjacency_matrix)

    """
    Format:
    True [True, True, True, True, True, True] 1
    True [True, False, False, False, False, True] 0
    False [False, False, False, False, False, False] -1
    False [False, False, False, False, False, False] -1
    """
    def find_differences(self):
        for pre_neuron in self.all_neurons:
            for post_neuron in self.all_neurons:
                interests_connections = [_matrix.loc[pre_neuron][post_neuron] for _matrix in self.interest_matrices]
                interests_connections = [_value for _value in interests_connections if str(_value) != 'nan']
                interests_flags = [interest_connection >= CONNECTION_THRESHOLD
                                   for interest_connection in interests_connections]
                interests_flag = decide_flag(flags_list=interests_flags)  # +1: all true, -1: all false, 0: mixed
                # our_connection = self.matrix_of_interest.loc[pre_neuron][post_neuron]
                # our_flag = our_connection >= CONNECTION_THRESHOLD  # connection exists

                others_connections = [_matrix.loc[pre_neuron][post_neuron] for _matrix in self.other_matrices]
                others_connections = [_value for _value in others_connections if str(_value) != 'nan']
                others_flags = [other_connection >= CONNECTION_THRESHOLD
                                for other_connection in others_connections]
                others_flag = decide_flag(flags_list=others_flags)  # +1: all true, -1: all false, 0: mixed

                if interests_flag == 1 and others_flag == -1:
                    self.added_connections.append((pre_neuron, post_neuron))
                elif others_flag == 1 and interests_flag == -1:
                    self.removed_connections.append((pre_neuron, post_neuron))

    def analyze_neurons(self):
        neurons_changes = {}
        for _neuron in self.all_neurons:
            neurons_changes[_neuron] = {}
        for added_connection in self.added_connections:
            pre_neuron, post_neuron = added_connection[0], added_connection[1]
            neurons_changes[pre_neuron][post_neuron] = 1
        for removed_connection in self.removed_connections:
            pre_neuron, post_neuron = removed_connection[0], removed_connection[1]
            neurons_changes[pre_neuron][post_neuron] = -1
        for _neuron in self.all_neurons:
            if not bool(neurons_changes[_neuron]):
                del neurons_changes[_neuron]
        # ############# Printing neurons_changes ############
        for pre_neuron, changes in neurons_changes.items():
            print(pre_neuron, changes)
        # ###################################################

    def build_connected_components(self):
        self.added_connected_components = adjacency_list_to_connected_components(
            adjacency_list=self.added_connections)
        self.removed_connected_components = adjacency_list_to_connected_components(
            adjacency_list=self.removed_connections)

    def get_added_connections(self):
        return self.added_connections

    def get_removed_connections(self):
        return self.removed_connections

    def get_added_connected_components(self):
        return self.added_connected_components

    def get_removed_connected_components(self):
        return self.removed_connected_components

    def find_components_edges(self):
        print(self.added_connections)
        print(len(self.added_connections))
        print(self.added_connected_components)
        print(len(self.added_connected_components))
        print(self.removed_connections)
        print(len(self.removed_connections))
        print(self.removed_connected_components)
        print(len(self.removed_connected_components))

        component_num = 0
        for connected_component in self.added_connected_components:
            component_edges = extract_connections(connected_component=connected_component,
                                                  list_of_edges=self.added_connections)
            self.added_components_edges.append(component_edges)
            component_num += 1

        component_num = 0
        for connected_component in self.removed_connected_components:
            component_edges = extract_connections(connected_component=connected_component,
                                                  list_of_edges=self.removed_connections)
            self.removed_components_edges.append(component_edges)
            component_num += 1

    def get_added_components_edges(self):
        return self.added_components_edges

    def get_removed_components_edges(self):
        return self.removed_components_edges

    def visualise_components(self, folder_to_save: str) -> None:
        component_num = 0
        for component_edges in self.added_components_edges:
            draw_graph(list_of_edges=component_edges,
                       saving_filepath=folder_to_save + 'added_component_' + str(component_num) + '.png')
            component_num += 1

        component_num = 0
        for component_edges in self.removed_components_edges:
            draw_graph(list_of_edges=component_edges,
                       saving_filepath=folder_to_save + 'removed_component_' + str(component_num) + '.png')
            component_num += 1


def get_cytoscape_weights():
    l3_filepath = './Adjacency Mats/Project_125.csv'
    dauer_filepath = './Adjacency Mats/Project_186.csv'
    l3_matrix = pd.read_csv(l3_filepath, index_col=[0])
    dauer_matrix = pd.read_csv(dauer_filepath, index_col=[0])
    all_neurons = l3_matrix.columns
    final_info = {}
    for _pre in all_neurons:
        final_info[_pre] = {}
        for _post in all_neurons:
            l3_synapse = l3_matrix[_post].loc[_pre]
            dauer_synapse = dauer_matrix[_post].loc[_pre]
            if str(dauer_synapse) == 'nan' or str(l3_synapse) == 'nan':
                _val = 0
            else:
                _val = dauer_synapse - l3_synapse
            if _val <= 0:
                final_info[_pre][_post] = -_val
            else:
                final_info[_pre][_post] = 0
    # with open('./dauer_positive.json', 'w') as file:
    #     json.dump(final_info, file)
    with open('./dauer_negative.json', 'w') as file:
        json.dump(final_info, file)
