# This is a sample Python script.


# from os import listdir
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import PymaidInterface
import AdjComparison
import SynapseDistribution
import NeuronsClassification
import ContactDistribution


def extract_adjacency_matrices():
    project_ids = [98, 106, 121, 124, 125, 133, 135, 161, 186, 266]
    for project_id in project_ids:
        project_catmaid = PymaidInterface.Skeleton(project_id=project_id)
        project_catmaid.make_adjacency_matrix()
        # project_catmaid.save_adjacency_matrix(filepath='./Adjacency_Mats.xlsx')
        filepath = './Adjacency Mats/Project_' + str(project_id) + '.csv'
        project_catmaid.save_adjacency_matrix(filepath=filepath)


def merge_left_and_right_adjacency_matrices():
    project_ids = [98, 106, 121, 124, 125, 133, 135, 161, 186, 266]
    for project_id in project_ids:
        adjacency_matrix_filepath = './Adjacency Mats/Project_' + str(project_id) + '.csv'
        small_adj_filepath_to_save = './Short Adj Mats/Project_' + str(project_id) + '.csv'
        PymaidInterface.merge_left_and_right(adjacency_matrix_filepath=adjacency_matrix_filepath,
                                             filepath_to_save=small_adj_filepath_to_save)


def analyze_adjacency_matrices():
    # matrices_folder = './Adjacency Mats/'
    matrices_folder = './Short Adj Mats/'
    # all_files = listdir(matrices_folder)
    # all_files.remove('Project_186.csv')
    interest_matrices_files = ['Project_266.csv', 'Project_186.csv']
    other_matrices_file = ['Project_98.csv', 'Project_106.csv', 'Project_121.csv', 'Project_124.csv',
                           'Project_125.csv', 'Project_133.csv', 'Project_135.csv', 'Project_161.csv']
    adjacency_comparator = AdjComparison.AdjacencyComparator(
        interest_matrices_filepath=[join(matrices_folder, file) for file in interest_matrices_files],
        other_matrices_filepath=[join(matrices_folder, file) for file in other_matrices_file])
    adjacency_comparator.find_differences()
    # adjacency_comparator.analyze_neurons()
    adjacency_comparator.build_connected_components()
    adjacency_comparator.find_components_edges()
    print(adjacency_comparator.get_added_components_edges())
    print(adjacency_comparator.get_removed_components_edges())
    # adjacency_comparator.visualise_components(folder_to_save='./Dauers_Plots/')


def plot_neuronal_components():
    # project_ids = [98, 106, 121, 124, 125, 133, 135, 161, 186, 266]
    # connected_component = ['ALA', 'ASK', 'ADL', 'OLL', 'URB', 'IL1V', 'SMDV', 'URAV', 'RMDD', 'RMEL/R', 'ASH',
    #                        'BDU', 'RIM', 'ADF', 'AVJ', 'RIF', 'AIM', 'RIP', 'AVH', 'IL1D', 'RMH', 'ASJ', 'URYV',
    #                        'AVA', 'OLQD', 'RMDV', 'PVQ', 'AWA', 'OLQV', 'AVE', 'RIB']  # ToDo: automatic
    component_edges = [('URAV', 'RMDD'), ('ALA', 'ASK'), ('ALA', 'ADL'), ('OLL', 'URB'), ('ASH', 'BDU'),
                       ('IL1V', 'URB'), ('IL1V', 'SMDV'), ('URYV', 'AVA'), ('ADF', 'AVJ'), ('ADF', 'RIF'),
                       ('ADF', 'AIM'), ('ADF', 'RIP'), ('BDU', 'RIM'), ('BDU', 'RIP'), ('AVH', 'RIP'), ('URB', 'URAV'),
                       ('URB', 'RMEL/R'), ('URB', 'AIM'), ('IL1D', 'URB'), ('IL1D', 'RMH'), ('ASK', 'ALA'),
                       ('ASK', 'AVJ'), ('ASK', 'ASJ'), ('AIM', 'AVA'), ('OLQD', 'OLL'), ('OLQD', 'RMDV'),
                       ('RMH', 'RMEL/R'), ('ASJ', 'AIM'), ('RMDV', 'URAV'), ('PVQ', 'BDU'), ('PVQ', 'RIP'),
                       ('AWA', 'RIP'), ('OLQV', 'OLL'), ('OLQV', 'AVE'), ('OLQV', 'RIB')]
    project_ids = [266]
    for project_id in project_ids:
        project_catmaid = PymaidInterface.Skeleton(project_id=project_id)
        project_catmaid.plot_connectors(list_of_pairs=component_edges)
        # project_catmaid.plot_component(list_of_neurons=connected_component)


def plot_single_neuron():
    PymaidInterface.draw_aligned_neuron(neuron_name='IL2V')
    # neuron_morphology = PymaidInterface.NeuronMorphology(neuron_name='ASJ', project_id=266)
    # neuron_morphology.find_skeleton()


def analyse_synapses():
    """
    This function, gets mean-synapses from SynapseDistribution.AdjacencyAnalyzer,
        and returns them
    Also, we can use adj_analyzer.plot_in_and_out_synapses() to plot input and output synapses together
        (x: input synapses, y: output synapses)
        Or adj_analyzer.plot_histograms() to plot histogram of synapses (neurons sorted based on num synapses)
    :return: mean_synapses
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Synapses distribution')

    adj_analyzer = SynapseDistribution.AdjacencyAnalyzer(adjacency_matrix_filepath='./Short Adj Mats/Project_186.csv')
    # adj_analyzer.plot_in_and_out_synapses()
    normalized_synapse_send, normalized_synapse_receive = adj_analyzer.plot_histograms()
    # non_zeros_send = normalized_synapse_send[normalized_synapse_send != 0.0]
    # non_zeros_receive = normalized_synapse_receive[normalized_synapse_receive != 0.0]

    # non_zero_send.sort_values().plot.bar(ax=ax1, subplots=True)
    normalized_synapse_send.sort_values().plot.bar(ax=ax1, subplots=True)
    ax1.set_title("Send")
    # non_zero_received.sort_values().plot.bar(ax=ax2, subplots=True)
    normalized_synapse_receive.sort_values().plot.bar(ax=ax2, subplots=True)
    ax2.set_title("Receive")

    plt.show()


def classify_neurons():
    neurons_classifier = NeuronsClassification.NeuronsClassifier(
        adjacency_matrix_filepath='./Short Adj Mats/Project_186.csv')


def analyse_contacts():
    """
    This function, gets mean-contacts of each dataset from ContactDistribution.ContactAnalyzer,
        and returns them
    :return: mean_contacts
    """
    contact_analyser = ContactDistribution.ContactAnalyzer(contactome_filepath='../Contactome/contactome.csv')
    normalized_class_contacts = contact_analyser.plot_histograms(matrix_index=5)
    # normalized_non_zero = normalized_class_contacts[normalized_class_contacts != 0.0]

    fig, ax = plt.subplots()
    fig.suptitle('Contact distribution')
    # normalized_non_zero.sort_values().plot.bar(ax=ax, subplots=True)
    normalized_class_contacts.plot.bar(ax=ax, subplots=True)
    ax.set_title("Net Surface Area")
    plt.show()


def synapses_and_contacts_trends():
    """
    This function, plots mean-synapses and mean-contacts in datasets together
    :return: A plot :)
    """
    matrices_folder = './Short Adj Mats/'
    # file_names = [124, 161, 121, 98, 106, 125, 133, 135, 186, 266]
    # labels = ['L1(0h)', 'L1(5h)', 'L1(8h)', 'L1(16h)', 'L2', 'L3', 'Ad-T', 'Ad-S', 'daf-2', 'dauer']
    # file_names = [124, 161, 121, 98, 106, 125, 135]
    file_names = [124, 161, 121, 98, 106, 125, 133, 135]
    # timepoints = [0, 5, 8, 16, 23, 27, 50]  # Connectome vs Contactome
    timepoints = [0, 5, 8, 16, 23, 27, 50, 50]
    # labels = ['L1(0h)', 'L1(5h)', 'L1(8h)', 'L1(16h)', 'L2', 'L3', 'Ad-S']  # Connectome vs Contactome
    labels = ['L1(0h)', 'L1(5h)', 'L1(8h)', 'L1(16h)', 'L2', 'L3', 'Ad-T', 'Ad-S']
    # matrices_file = ['Project_124.csv', 'Project_161.csv', 'Project_121.csv', 'Project_98.csv', 'Project_106.csv',
    #                  'Project_125.csv', 'Project_133.csv', 'Project_135.csv', 'Project_266.csv', 'Project_186.csv']
    mean_synapses = []
    for file_name in file_names:
        matrix_filepath = join(matrices_folder, 'Project_' + str(file_name) + '.csv')
        adj_analyzer = SynapseDistribution.AdjacencyAnalyzer(adjacency_matrix_filepath=matrix_filepath)
        mean_synapses.append(adj_analyzer.get_mean_synapses())

    del mean_synapses[3]
    del labels[3]
    del timepoints[3]

    synapses_model = LinearRegression()
    synapses_model.fit(np.array(timepoints).reshape(-1, 1), np.array(mean_synapses))
    syn_scale, syn_shift = synapses_model.coef_[0], synapses_model.intercept_
    # ########################## Contactome ############################
    # contact_analyser = ContactDistribution.ContactAnalyzer(contactome_filepath='../Contactome/contactome.csv')
    # mean_contacts = contact_analyser.get_mean_synapses()
    # del mean_contacts[3]
    #
    # contacts_model = LinearRegression()
    # contacts_model.fit(np.array(timepoints).reshape(-1, 1), np.array(mean_contacts))
    # con_scale, con_shift = contacts_model.coef_[0], contacts_model.intercept_
    #
    # normalization_coeff = syn_scale / con_scale
    # normalized_contacts = [(_contact - con_shift) * normalization_coeff + syn_shift
    #                        for _contact in mean_contacts]

    # ################# Dauer #####################3
    matrix_filepath = join(matrices_folder, 'Project_186.csv')
    adj_analyzer = SynapseDistribution.AdjacencyAnalyzer(adjacency_matrix_filepath=matrix_filepath)
    daf2 = adj_analyzer.get_mean_synapses()

    # matrix_filepath = join(matrices_folder, 'Project_266.csv')
    # adj_analyzer = SynapseDistribution.AdjacencyAnalyzer(adjacency_matrix_filepath=matrix_filepath)
    # dauer = adj_analyzer.get_mean_synapses()
    # print(daf2, dauer)
    # ################### Plot ####################
    fig, ax = plt.subplots()
    ax.scatter(timepoints, mean_synapses)
    for _ind, _label in enumerate(labels):
        ax.annotate(_label, (timepoints[_ind], mean_synapses[_ind]))

    ax.plot(timepoints, [syn_shift + syn_scale * _x for _x in timepoints], linestyle='dashed')
    ax.plot(timepoints, [daf2 for _x in timepoints], label='daf-2')
    plt.legend(loc="upper left")
    plt.xlim(-3, 54)
    # plt.ylim(2, 7)

    # ax.scatter(timepoints, normalized_contacts)
    # for _ind, _label in enumerate(labels):
    #     ax.annotate(_label, (timepoints[_ind], normalized_contacts[_ind]))

    # plt.title("Connectome vs Contactome")
    plt.title("Total number of synapses")
    plt.xlabel("Time (hr)")
    # plt.ylabel("Synapse (#) and\nSurface area (" + str(round(1 / normalization_coeff)) + " nm2)")
    plt.ylabel("#Synapse")
    plt.show()


def compare_synapse_vs_contact_dist():
    _ind = 5
    dataset_filepath = './Short Adj Mats/Project_125.csv'
    adj_analyzer = SynapseDistribution.AdjacencyAnalyzer(adjacency_matrix_filepath=dataset_filepath)
    # normalized_synapse_send, normalized_synapse_receive = adj_analyzer.plot_histograms()
    normalized_pairs = adj_analyzer.plot_histograms()

    contact_analyser = ContactDistribution.ContactAnalyzer(contactome_filepath='../Contactome/contactome.csv')
    normalized_class_contacts = contact_analyser.plot_histograms(matrix_index=_ind)
    # normalized_connections = pd.concat([normalized_synapse_send, normalized_synapse_receive, normalized_class_contacts],
    #                                    axis=1,
    #                                    keys=['Send_syn', 'Receive_syn', 'Contact'])
    normalized_connections = pd.concat([normalized_pairs, normalized_class_contacts],
                                       axis=1,
                                       keys=['Synapse', 'Contact'])
    # normalized_connections['Synapse'] = normalized_connections['Send_syn'].div(2) + normalized_connections['Receive_syn'].div(2)
    normalized_connections['Efficiency'] = normalized_connections['Synapse'] / normalized_connections['Contact'] / 8
    # sorted_connections = normalized_connections.sort_values(by=['Synapse', 'Contact'])
    # sorted_connections = normalized_connections.sort_values(by=['Contact', 'Synapse'])
    sorted_connections = normalized_connections.sort_values(by=['Efficiency'])
    # Filters:
    sorted_connections = sorted_connections.loc[(sorted_connections != 0).all(axis=1)]
    # sorted_connections = sorted_connections.loc[sorted_connections['Synapse'] > 0.1]

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig.suptitle('Synapses and Contact distribution')
    # sorted_connections.plot.bar(ax=ax1, subplots=True)

    plt.gcf().subplots_adjust(bottom=0.15)
    fig = plt.figure()
    ax = sorted_connections.tail(30).plot(y=["Synapse", "Contact", "Efficiency"], kind="bar", ax=fig.gca())
    ax.set_xlabel('Neurons', fontsize=10)
    # normalized_synapse_send.plot.bar(ax=ax1, subplots=True)
    # ax1.set_title("Send")
    # normalized_synapse_receive.plot.bar(ax=ax2, subplots=True)
    # ax2.set_title("Receive")
    # normalized_class_contacts.plot.bar(ax=ax3, subplots=True)
    # ax3.set_title("Net Surface Area")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':  # Todo: descriptions
    # extract_adjacency_matrices()
    # merge_left_and_right_adjacency_matrices()
    # analyze_adjacency_matrices()
    # plot_neuronal_components()
    plot_single_neuron()
    # analyse_synapses()
    # analyse_contacts()
    # compare_synapse_vs_contact_dist()
    # classify_neurons()
    # synapses_and_contacts_trends()
    # AdjComparison.get_cytoscape_weights()
