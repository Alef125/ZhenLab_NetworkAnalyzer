import dijkstar.algorithm
import pandas as pd
from dijkstar import Graph, find_path
# import numpy as np
# import matplotlib.pyplot as plt


SENSORY_NEURONS = ['ADF', 'ADL', 'AFD', 'ALM', 'ALN', 'AQR', 'ASE', 'ASG', 'ASH', 'ASI',
                   'ASJ', 'ASK', 'AUA', 'AVM', 'AWA', 'AWB', 'AWC', 'BAG', 'DVA', 'FLP',
                   # 'IL2', 'OLQ', 'PHA', 'PHB', 'PHC', 'PLM', 'PQR', 'PVD', 'PVM', 'SAA', 'URY'
                   'OLL', 'PLN', 'SDQ', 'URB', 'URX']

MODULATORY_NEURONS = ['BWM01V', 'BWM02V', 'BWM03V', 'BWM04V', 'BWM05V', 'BWM06V', 'BWM07V', 'BWM08V',
                      'BWMD01D', 'BWMD02D', 'BWMD03D', 'BWMD04D', 'BWMD05D', 'BWMD06D', 'BWMD07D', 'BWMD08D']


class NeuronsClassifier:
    def __init__(self, adjacency_matrix_filepath: str):
        self.adjacency_matrix = pd.read_csv(adjacency_matrix_filepath, index_col=[0])
        self.all_neurons = self.adjacency_matrix.columns.tolist()
        self.add_source_and_sink()
        self.adjacency_graph = Graph()
        self.build_adjacency_graph()
        # ##############################
        self.processing_depths = dict(zip(self.all_neurons, [0] * len(self.all_neurons)))
        self.calculate_processing_depths()
        # ###########################################################

    def add_source_and_sink(self):
        self.adjacency_matrix['Source'] = 0
        self.adjacency_matrix['Sink'] = 0
        self.adjacency_matrix.loc['Source'] = 0
        self.adjacency_matrix.loc['Sink'] = 0
        for sensory_neuron in SENSORY_NEURONS:
            self.adjacency_matrix[sensory_neuron].loc['Source'] = 1
        for modulatory_neuron in MODULATORY_NEURONS:
            self.adjacency_matrix['Sink'].loc[modulatory_neuron] = 1

    def build_adjacency_graph(self):
        all_nodes = self.all_neurons.copy()
        all_nodes.append('Source')
        all_nodes.append('Sink')
        num_nodes = len(all_nodes)
        for node_1 in range(num_nodes):
            _neuron_1 = all_nodes[node_1]
            for node_2 in range(num_nodes):
                _neuron_2 = all_nodes[node_2]
                connection_strength = self.adjacency_matrix[_neuron_2].loc[_neuron_1]
                if connection_strength > 0:
                    self.adjacency_graph.add_edge(node_1, node_2, 1)

    def calculate_processing_depths(self):
        source_id = len(self.all_neurons)
        for _neuron_id in range(len(self.all_neurons)):
            _neuron = self.all_neurons[_neuron_id]
            try:
                shortest_path = find_path(self.adjacency_graph, source_id, _neuron_id)
                self.processing_depths[_neuron] = shortest_path.total_cost  # distance from Source to _neuron
            except dijkstar.algorithm.NoPathError:
                print('No path from source to ' + _neuron)
                self.processing_depths[_neuron] = None
        print(self.processing_depths)
