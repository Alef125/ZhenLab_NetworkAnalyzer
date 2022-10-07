"""
This script, provides classes and functions to receive data from CATMAID

classes:
Skeleton,
NeuronMorphology,
PointTransformer,
CatmaidProject
"""


import pymaid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# import TdAnnotation
# from mpl_toolkits import mplot3d
# from openpyxl import load_workbook

ALL_NEURONS = [['ADAL', 'ADAR'], ['ADEL', 'ADER'], ['ADFL', 'ADFR'], ['ADLL', 'ADLR'], ['AFDL', 'AFDR'],
               ['AIAL', 'AIAR'], ['AIBL', 'AIBR'], ['AIML', 'AIMR'], ['AINL', 'AINR'], ['AIYL', 'AIYR'],
               ['AIZL', 'AIZR'], ['ALA'], ['ALML', 'ALMR'], ['ALNL', 'ALNR'], ['AQR'], ['ASEL', 'ASER'],
               ['ASGL', 'ASGR'], ['ASHL', 'ASHR'], ['ASIL', 'ASIR'], ['ASJL', 'ASJR'], ['ASKL', 'ASKR'],
               ['AUAL', 'AUAR'], ['AVAL', 'AVAR'], ['AVBL', 'AVBR'], ['AVDL', 'AVDR'], ['AVEL', 'AVER'],
               ['AVFL', 'AVFR'], ['AVHL', 'AVHR'], ['AVJL', 'AVJR'], ['AVKL', 'AVKR'], ['AVL'], ['AVM'],
               ['AWAL', 'AWAR'], ['AWBL', 'AWBR'], ['AWCL', 'AWCR'], ['BAGL', 'BAGR'], ['BDUL', 'BDUR'],
               ['CEPDL', 'CEPDR'], ['CEPVL', 'CEPVR'], ['DVA'], ['DVC'], ['FLPL', 'FLPR'], ['IL1DL', 'IL1DR'],
               ['IL1L', 'IL1R'], ['IL1VL', 'IL1VR'], ['IL2DL', 'IL2DR'], ['IL2L', 'IL2R'], ['IL2VL', 'IL2VR'],
               ['OLLL', 'OLLR'], ['OLQDL', 'OLQDR'], ['OLQVL', 'OLQVR'], ['PLNL', 'PLNR'], ['PVCL', 'PVCR'],
               ['PVNL', 'PVNR'], ['PVPL', 'PVPR'], ['PVQL', 'PVQR'], ['PVR'], ['PVT'], ['RIAL', 'RIAR'],
               ['RIBL', 'RIBR'], ['RICL', 'RICR'], ['RID'], ['RIFL', 'RIFR'], ['RIGL', 'RIGR'], ['RIH'],
               ['RIML', 'RIMR'], ['RIPL', 'RIPR'], ['RIR'], ['RIS'], ['RIVL', 'RIVR'], ['RMDDL', 'RMDDR'],
               ['RMDVL', 'RMDVR'], ['RMDL', 'RMDR'], ['RMED', 'RMEV'], ['RMEL', 'RMER'], ['RMFL', 'RMFR'],
               ['RMGL', 'RMGR'], ['RMHL', 'RMHR'], ['SAADL', 'SAADR'], ['SAAVL', 'SAAVR'], ['SDQL', 'SDQR'],
               ['SIADL', 'SIADR'], ['SIAVL', 'SIAVR'], ['SIBDL', 'SIBDR'], ['SIBVL', 'SIBVR'], ['SMBDL', 'SMBDR'],
               ['SMBVL', 'SMBVR'], ['SMDDL', 'SMDDR'], ['SMDVL', 'SMDVR'], ['URADL', 'URADR'], ['URAVL', 'URAVR'],
               ['URBL', 'URBR'], ['URXL', 'URXR'], ['URYDL', 'URYDR'], ['URYVL', 'URYVR'], ["BWM-VL01", "BWM-VR01"],
               ["BWM-DL01", "BWM-DR01"], ["BWM-VL02", "BWM-VR02"], ["BWM-DL02", "BWM-DR02"], ["BWM-VL03", "BWM-VR03"],
               ["BWM-DL03", "BWM-DR03"], ["BWM-VL04", "BWM-VR04"], ["BWM-DL04", "BWM-DR04"], ["BWM-VL05", "BWM-VR05"],
               ["BWM-DL05", "BWM-DR05"], ["BWM-VL06", "BWM-VR06"], ["BWM-DL06", "BWM-DR06"], ["BWM-VL07", "BWM-VR07"],
               ["BWM-DL07", "BWM-DR07"], ["BWM-VL08", "BWM-VR08"], ["BWM-DL08", "BWM-DR08"], ['GLRVR', 'GLRVL'],
               ['GLRDL', 'GLRDR'], ["GLRL", "GLRR"], ["CEPshVL", "CEPshVR"], ["CEPshDL", "CEPshDR"],
               ["HSNL", "HSNR"], ["CANL", "CANR"]]  # 224 neurons

# DIRECTION_TRANSLATION = {'upstream': -1, 'downstream': 1}
# List of neurons / other cell types which we exclude from our analysis
EXCLUSION_LIST = ['fragment', 'small_fragment', 'large fragment', 'large unknown fragment', 'tiny_fragment',
                  'fragment_outside_nervering', 'fragment_in_ventral_NR', 'fragment_in_ventral_nr',
                  'sheath_glia', 'Exc gl', 'exc gl', 'exc_gl', 'w_synapse', 'mu', 'VA1', 'VA2', 'VB1',  'VB2',
                  'VD1', 'VD2', 'AS1', 'AS2', 'SABD', 'DA1', 'DB1', 'DD1', 'SABVR', 'SABVL', 'RVG_n_posterior3 [VA4?]',
                  'Neuron 2610110', 'Neuron 2610344', 'Neuron 2608001',
                  'NR_fragment', 'AMshR', 'PVNL_or_R', 'XXXR', 'VNC_fragment',
                  'PVDL', 'PVDR', 'glia', 'muscle', 'muscle_fragment', 'PVNL/R', 'BWM-DL',
                  'muscle fragment 1', 'muscle fragment 2', 'muscle fragment 3', 'muscle fragment 4',
                  'muscle fragment 5', 'muscle fragment 6', 'muscle fragment 7', 'muscle fragment 8',
                  'muscle fragment 9', 'muscle fragment 10', 'muscle fragment 11', 'muscle fragment 12',
                  'muscle fragment 13', 'muscle fragment 14', 'muscle fragment 15',  'muscle fragment 16',
                  'muscle fragment 17', 'muscle fragment 18',
                  'hyp', 'DB3', 'DB5', 'DD2', 'DD3', 'exc canal L', 'NSML', 'NSMR', 'AMshL', 'BWM-DL14', 'BWM-DR14']

# Class (pair) for each neuron (mainly are removing L/R from the name)
LR_DICT = {'ADAL': 'ADA', 'ADAR': 'ADA', 'ADEL': 'ADE', 'ADER': 'ADE', 'ADFL': 'ADF', 'ADFR': 'ADF',
           'ADLL': 'ADL', 'ADLR': 'ADL', 'AFDL': 'AFD', 'AFDR': 'AFD', 'AIAL': 'AIA', 'AIAR': 'AIA',
           'AIBL': 'AIB', 'AIBR': 'AIB', 'AIML': 'AIM', 'AIMR': 'AIM', 'AINL': 'AIN', 'AINR': 'AIN',
           'AIYL': 'AIY', 'AIYR': 'AIY', 'AIZL': 'AIZ', 'AIZR': 'AIZ', 'ALA': 'ALA', 'ALML': 'ALM', 'ALMR': 'ALM',
           'ALNL': 'ALN', 'ALNR': 'ALN', 'AQR': 'AQR', 'ASEL': 'ASE', 'ASER': 'ASE', 'ASGL': 'ASG', 'ASGR': 'ASG',
           'ASHL': 'ASH', 'ASHR': 'ASH', 'ASIL': 'ASI', 'ASIR': 'ASI', 'ASJL': 'ASJ', 'ASJR': 'ASJ',
           'ASKL': 'ASK', 'ASKR': 'ASK', 'AUAL': 'AUA', 'AUAR': 'AUA', 'AVAL': 'AVA', 'AVAR': 'AVA',
           'AVBL': 'AVB', 'AVBR': 'AVB', 'AVDL': 'AVD', 'AVDR': 'AVD', 'AVEL': 'AVE', 'AVER': 'AVE',
           'AVFL': 'AVF', 'AVFR': 'AVF', 'AVHL': 'AVH', 'AVHR': 'AVH', 'AVJL': 'AVJ', 'AVJR': 'AVJ',
           'AVKL': 'AVK', 'AVKR': 'AVK', 'AVL': 'AVL', 'AVM': 'AVM', 'AWAL': 'AWA', 'AWAR': 'AWA',
           'AWBL': 'AWB', 'AWBR': 'AWB', 'AWCL': 'AWC', 'AWCR': 'AWC', 'BAGL': 'BAG', 'BAGR': 'BAG',
           'BDUL': 'BDU', 'BDUR': 'BDU', 'CEPDL': 'CEPD', 'CEPDR': 'CEPD', 'CEPVL': 'CEPV', 'CEPVR': 'CEPV',
           'DVA': 'DVA', 'DVC': 'DVC', 'FLPL': 'FLP', 'FLPR': 'FLP',
           'IL1DL': 'IL1D', 'IL1DR': 'IL1D', 'IL1L':  'IL1L/R', 'IL1R':  'IL1L/R', 'IL1VL': 'IL1V', 'IL1VR': 'IL1V',
           'IL2DL': 'IL2D', 'IL2DR': 'IL2D', 'IL2L': 'IL2L/R', 'IL2R': 'IL2L/R', 'IL2VL': 'IL2V', 'IL2VR': 'IL2V',
           'OLLL': 'OLL', 'OLLR': 'OLL', 'OLQDL': 'OLQD', 'OLQDR': 'OLQD', 'OLQVL': 'OLQV', 'OLQVR': 'OLQV',
           'PLNL': 'PLN', 'PLNR': 'PLN', 'PVCL': 'PVC', 'PVCR': 'PVC', 'PVNL': 'PVN', 'PVNR': 'PVN',
           'PVPL': 'PVP', 'PVPR': 'PVP', 'PVQL': 'PVQ', 'PVQR': 'PVQ', 'PVR': 'PVR', 'PVT': 'PVT',
           'RIAL': 'RIA', 'RIAR': 'RIA', 'RIBL': 'RIB', 'RIBR': 'RIB', 'RICL': 'RIC', 'RICR': 'RIC',
           'RID': 'RID', 'RIFL': 'RIF', 'RIFR': 'RIF', 'RIGL': 'RIG', 'RIGR': 'RIG', 'RIH': 'RIH',
           'RIML': 'RIM', 'RIMR': 'RIM', 'RIPL': 'RIP', 'RIPR': 'RIP', 'RIR': 'RIR', 'RIS': 'RIS',
           'RIVL': 'RIV', 'RIVR': 'RIV',
           'RMDDL': 'RMDD', 'RMDDR': 'RMDD', 'RMDVL': 'RMDV', 'RMDVR': 'RMDV', 'RMDL': 'RMDL/R', 'RMDR': 'RMDL/R',
           'RMED': 'RMED', 'RMEV': 'RMEV', 'RMEL': 'RMEL/R', 'RMER': 'RMEL/R',
           'RMFL': 'RMF', 'RMFR': 'RMF', 'RMGL': 'RMG', 'RMGR': 'RMG', 'RMHL': 'RMH', 'RMHR': 'RMH',
           'SAADL': 'SAAD', 'SAADR': 'SAAD', 'SAAVL': 'SAAV', 'SAAVR': 'SAAV', 'SDQL': 'SDQ', 'SDQR': 'SDQ',
           'SIADL': 'SIAD', 'SIADR': 'SIAD', 'SIAVL': 'SIAV', 'SIAVR': 'SIAV', 'SIBDL': 'SIBD', 'SIBDR': 'SIBD',
           'SIBVL': 'SIBV', 'SIBVR': 'SIBV', 'SMBDL': 'SMBD', 'SMBDR': 'SMBD', 'SMBVL': 'SMBV', 'SMBVR': 'SMBV',
           'SMDDL': 'SMDD', 'SMDDR': 'SMDD', 'SMDVL': 'SMDV', 'SMDVR': 'SMDV', 'URADL': 'URAD', 'URADR': 'URAD',
           'URAVL': 'URAV', 'URAVR': 'URAV', 'URBL': 'URB', 'URBR': 'URB', 'URXL': 'URX', 'URXR': 'URX',
           'URYDL': 'URYD', 'URYDR': 'URYD', 'URYVL': 'URYV', 'URYVR': 'URYV',
           "BWM-VL01": 'BWM01V', "BWM-VR01": 'BWM01V', "BWM-DL01": 'BWMD01D', "BWM-DR01": 'BWMD01D',
           "BWM-VL02": 'BWM02V', "BWM-VR02": 'BWM02V', "BWM-DL02": 'BWMD02D', "BWM-DR02": 'BWMD02D',
           "BWM-VL03": 'BWM03V', "BWM-VR03": 'BWM03V', "BWM-DL03": 'BWMD03D', "BWM-DR03": 'BWMD03D',
           "BWM-VL04": 'BWM04V', "BWM-VR04": 'BWM04V', "BWM-DL04": 'BWMD04D', "BWM-DR04": 'BWMD04D',
           "BWM-VL05": 'BWM05V', "BWM-VR05": 'BWM05V', "BWM-DL05": 'BWMD05D', "BWM-DR05": 'BWMD05D',
           "BWM-VL06": 'BWM06V', "BWM-VR06": 'BWM06V', "BWM-DL06": 'BWMD06D', "BWM-DR06": 'BWMD06D',
           "BWM-VL07": 'BWM07V', "BWM-VR07": 'BWM07V', "BWM-DL07": 'BWMD07D', "BWM-DR07": 'BWMD07D',
           "BWM-VL08": 'BWM08V', "BWM-VR08": 'BWM08V', "BWM-DL08": 'BWMD08D', "BWM-DR08": 'BWMD08D',
           'GLRVR': 'GLRV', 'GLRVL': 'GLRV', 'GLRDL': 'GLRD', 'GLRDR': 'GLRD', "GLRL": 'GLRL+R', "GLRR": 'GLRL+R',
           "CEPshVL": 'CEPshV', "CEPshVR": 'CEPshV', "CEPshDL": 'CEPshD', "CEPshDR": 'CEPshD',
           "HSNL": 'HSN', "HSNR": 'HSN', "CANL": 'CAN', "CANR": 'CAN'}

# Neurons used in alignment
PIN_NEURONS = ['RID', 'AVAL', 'AVAR', 'RIPL', 'RIPR', 'RMEV', 'SMDDL', 'SMDDR', 'RMED', 'BAGL', 'BAGR']


def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    """
    return plt.cm.get_cmap(name, n)


def refine_neuron_name(raw_name: str) -> str:
    """
    This function, refines the neuron name.
        For instance, removing extra '.', ',', '[...]', '?', or '!' characters from the name
    :param raw_name: The neuron's raw name
    :return: refined name of the neuron
    """
    # processed_name = raw_name.replace('-', '')
    processed_name = raw_name
    if processed_name[0] == '[' and raw_name[-1] == ']':
        processed_name = processed_name[1:-1]
    if processed_name[-1] == '.':
        processed_name = processed_name[:-1]
    if processed_name[-1] == '!':
        processed_name = processed_name[:-1]
    if processed_name[-1] == '?':
        processed_name = processed_name[:-1]
    # Doing it twice!
    if processed_name[-1] == '?':
        processed_name = processed_name[:-1]
    return processed_name


def find_neuron_id(neuron_name: str) -> int:
    """
    This function, finds the CATMAID id of a neuron, using its name
    :param neuron_name: The neuron's name
    :return: The CATMAID id of the query neuron
    """
    try:
        all_found = pymaid.get_neuron(neuron_name)
    except Exception:
        return None
    if type(all_found) == pymaid.core.CatmaidNeuron:
        return all_found.id
    elif type(all_found) == pymaid.core.CatmaidNeuronList:
        for _neuron in all_found:
            if refine_neuron_name(raw_name=_neuron.name) == neuron_name:
                return _neuron.id


class Skeleton:  # Todo: merge with NeuronMorphology
    """
    This class, is used to store neurons' information in CATMAID to make adjacency matrix
        i.e., project_id --> adjacency matrix
    """
    def __init__(self, project_id: int):
        """
        :param project_id: id of a project
        """
        pymaid.CatmaidInstance(
            server='https://zhencatmaid.com/',
            api_token='c48243e19b85edf37345ced8049ce5d6c5802412',
            project_id=project_id
            # http_user='user',  # omit if not required
            # http_password='pw',  # omit if not required
        )
        self.project_id = project_id
        self.num_neurons = None
        self.neurons_index_map = {}  # Assigning an index from 0 to 243 to ALL_NEURONS
        self.neurons_reverse_index_map = {}  # Index to name
        self.adjacency_matrix = None
        self.make_neurons_index_map()

    def make_neurons_index_map(self):
        """
        This method, makes self.neurons_index_map and self.neurons_reverse_index_map
        :return: -
        """
        cnt = 0
        for _neuron_pair in ALL_NEURONS:
            for _neuron_name in _neuron_pair:
                self.neurons_index_map[_neuron_name] = cnt
                self.neurons_reverse_index_map[cnt] = _neuron_name
                cnt += 1
        self.num_neurons = cnt
        self.adjacency_matrix = np.zeros(shape=(cnt, cnt))

    def put_aside_neuron(self, neuron_name: str) -> None:
        """
        This method, inserts NaN in the adjacency matrix for non-existing neurons in this project
        :param neuron_name: Name of removing neuron
        :return: -
        """
        neuron_id = self.neurons_index_map[neuron_name]
        self.adjacency_matrix[neuron_id, :] = np.nan
        self.adjacency_matrix[:, neuron_id] = np.nan

    def make_adjacency_matrix(self):
        """
        This method, fills self.adjacency_matrix
        :return: -
        """
        for _neuron_pair in ALL_NEURONS:
            for _neuron1_name in _neuron_pair:
                _neuron_id = find_neuron_id(neuron_name=_neuron1_name)
                if not _neuron_id:
                    print("Neuron " + _neuron1_name + " is missing in this dataset")
                    self.put_aside_neuron(neuron_name=_neuron1_name)
                    continue
                _neuron_partners = pymaid.get_partners(_neuron_id)
                i_index = self.neurons_index_map[_neuron1_name]
                for _index, _row in _neuron_partners.iterrows():
                    # neuron_name, skeleton_id, num_nodes, relation, 2161601,  total
                    _neuron2_name = refine_neuron_name(_row['neuron_name'])
                    if _neuron2_name in EXCLUSION_LIST:
                        continue
                    try:
                        j_index = self.neurons_index_map[_neuron2_name]
                    except KeyError:
                        print('Exception', _neuron2_name, _row['neuron_name'])
                        continue
                    # connection_value = DIRECTION_TRANSLATION[_row['relation']] * _row['total']
                    if _row['relation'] == 'downstream':
                        if self.adjacency_matrix[i_index, j_index] != 0:
                            if _row['total'] != self.adjacency_matrix[i_index, j_index]:
                                print("Not symmetrical connections!")
                        self.adjacency_matrix[i_index, j_index] = _row['total']
                    else:  # 'upstream'
                        if self.adjacency_matrix[j_index, i_index] != 0:
                            if _row['total'] != self.adjacency_matrix[j_index, i_index]:
                                print("Not symmetrical connections!")
                        self.adjacency_matrix[j_index, i_index] = _row['total']

    def save_adjacency_matrix(self, filepath: str) -> None:
        """
        This method, saved the adjacency_matrix .csv file in filepath
        :param filepath: The filepath to save self.adjacency_matrix
        :return: -
        """
        all_neurons_names = list(self.neurons_index_map.keys())
        adjacency_matrix_df = pd.DataFrame(self.adjacency_matrix,
                                           index=all_neurons_names,
                                           columns=all_neurons_names)
        # sheet_name = "Project_" + str(self.project_id)
        adjacency_matrix_df.to_csv(filepath)
        """
        Excel saving:
        book = load_workbook(filepath)
        writer = pd.ExcelWriter(filepath, engine='openpyxl')
        writer.book = book
        adjacency_matrix_df.to_excel(writer, sheet_name=sheet_name)
        writer.save()
        writer.close()
        """
        print("Dataset " + str(self.project_id) + " Successfully saved")

    def plot_component(self, list_of_neurons: list):
        """
        This methods, plots the skeleton for the neurons in list_of_neurons
        :param list_of_neurons: List of neuron's names to be plotted
        :return: -
        """
        cnt = 0
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        cmap = get_cmap(len(list_of_neurons))
        for _neuron_id in list_of_neurons:
            try:
                _neuron_info = pymaid.get_neuron(_neuron_id).to_dataframe()
                # ['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags']
                neuron_skeleton = _neuron_info['nodes'][0]
                # ['node_id', 'parent_id', 'creator_id', 'x', 'y', 'z', 'radius', 'confidence', 'type']
                # ax.plot3D(neuron_skeleton['x'].tolist(),
                #           neuron_skeleton['y'].tolist(),
                #           neuron_skeleton['z'].tolist(),
                #           'gray')

                ax.scatter3D(neuron_skeleton['x'].tolist(),
                             neuron_skeleton['y'].tolist(),
                             neuron_skeleton['z'].tolist(),
                             c=cmap(cnt),  # to cnt
                             s=5,
                             # cmap='Reds'
                             label=_neuron_id)
                # plt.legend(loc='best')
                cnt += 1
                # plt.show()
            except Exception:
                pass
        plt.legend(list_of_neurons)
        plt.show()

    def plot_connectors(self, list_of_pairs: list):
        """
        This method, plots the 3D distribution of contactors (synapses)
            between neuronal pair inside list_of_pairs
        :param list_of_pairs: List of neuronal pairs: [(pre-synaptic, post-synaptic)]
        :return: -
        """
        x_list = []
        y_list = []
        z_list = []
        labels = []
        for _pair in list_of_pairs:
            pre_neuron_name, post_neuron_name = _pair[0], _pair[1]
            try:
                pre_neuron_info = pymaid.get_neuron(pre_neuron_name).to_dataframe()
                post_neuron_info = pymaid.get_neuron(post_neuron_name).to_dataframe()
                # ['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags']
                pre_neuron_ids = [int(_id) for _id in pre_neuron_info['skeleton_id'].tolist()]
                post_neuron_ids = [int(_id) for _id in post_neuron_info['skeleton_id'].tolist()]
                for pre_neuron_all_connectors in pre_neuron_info['connectors']:
                    # ['node_id', 'connector_id', 'type' (0 or 1), 'x', 'y', 'z']  Todo: synapse vs. gap junc.
                    for _index, _row in pre_neuron_all_connectors.iterrows():
                        _connector_id = int(_row['connector_id'])
                        _connector_info = pymaid.get_connector_details(_connector_id)
                        # ['connector_id', 'presynaptic_to', 'postsynaptic_to',
                        #   'presynaptic_to_node', 'postsynaptic_to_node']
                        connector_presynaptic_neuron = _connector_info['presynaptic_to'][0]
                        connector_postsynaptic_neurons = _connector_info['postsynaptic_to'][0]
                        pre_flag = connector_presynaptic_neuron in pre_neuron_ids
                        post_flag = not set(post_neuron_ids).isdisjoint(connector_postsynaptic_neurons)
                        if pre_flag and post_flag:
                            x_list.append(_row['x'])
                            y_list.append(_row['y'])
                            z_list.append(_row['z'])
                            labels.append(str(pre_neuron_name) + "," + str(post_neuron_name))
            except Exception:
                pass
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x_list, y_list, z_list,
                     c='r',  # to cnt
                     s=15)
        for _ind, _label in enumerate(labels):
            ax.text(x_list[_ind], y_list[_ind], z_list[_ind], '%s' % _label, size=5, zorder=1, color='k')
            # TdAnnotation.annotate3d(ax, s=_label, xyz=(x_list[_ind], y_list[_ind], z_list[_ind]))
            # ax.annotate(_label, (x_list[_ind], y_list[_ind], z_list[_ind]))
        plt.show()


def points_to_coordinates(list_of_points: list):
    """
    This function, separates x, y, z coordinates from a list of points in (x, y, z) format
    :param list_of_points: [(x, y, z)]
    :return: Separated coordinates
    """
    _xs, _ys, _zs = [], [], []
    for _point in list_of_points:
        _xs.append(_point[0])
        _ys.append(_point[1])
        _zs.append(_point[2])
    return _xs, _ys, _zs


def merge_left_and_right(adjacency_matrix_filepath: str, filepath_to_save: str) -> None:
    """
    This function gets the adjacency_matrix of full neurons, and makes the adjacency_matrix for classes
        (L/R pairs together)
    :param adjacency_matrix_filepath: Filepath for the full adjacency_matrix
    :param filepath_to_save: Filepath to save the shrunk adjacency_matrix
    :return: -
    """
    adjacency_matrix = pd.read_csv(adjacency_matrix_filepath, index_col=[0])
    all_neurons_names = adjacency_matrix.columns.tolist()
    abr_neurons_names = list(set(LR_DICT.values()))
    num_abr_neurons = len(abr_neurons_names)
    small_adj_matrix = pd.DataFrame(np.zeros(shape=(num_abr_neurons, num_abr_neurons)),
                                    index=abr_neurons_names,
                                    columns=abr_neurons_names)
    for pre_neuron in all_neurons_names:
        for post_neuron in all_neurons_names:
            abr_pre_neuron = LR_DICT[pre_neuron]
            abr_post_neuron = LR_DICT[post_neuron]
            our_connection = adjacency_matrix.loc[pre_neuron][post_neuron]
            small_adj_matrix.loc[abr_pre_neuron][abr_post_neuron] += our_connection
    small_adj_matrix.to_csv(filepath_to_save)


def get_neuron_name_from_id(neuron_id: int) -> str:
    """
    (Must be modified, as the usage of "pymaid" project_id is ambiguous)
    :param neuron_id: The id of the desired neuron
    :return: Name of the desired neuron
    """
    all_names = []
    _neuron_info = pymaid.get_neuron(neuron_id).to_dataframe()
    for _ind, _sub_neuron in _neuron_info.iterrows():  # Mainly L & R
        _sub_neuron_name = refine_neuron_name(_sub_neuron['neuron_name'])
        _sub_neuron_skeleton = _sub_neuron['nodes']
        if _sub_neuron_name[-2:] == '/R' or _sub_neuron_name[-2:] == '/L' or len(_sub_neuron_skeleton) < 50:
            continue
        else:
            all_names.append(_sub_neuron_name)
    return ''.join(str(s) for s in all_names)


def is_inside_nerve_ring(skeleton_tree: pd.DataFrame,
                         nerve_ring_starts: int,  # Node ID
                         nerve_ring_ends: int,  # Node ID
                         point_of_interest: int) -> bool:  # Node ID
    """

    :param skeleton_tree: Skeleton dataframe with (at least) node_id, parent_id, type columns
    :param nerve_ring_starts: The node ID for the starting point
    :param nerve_ring_ends: The node ID for the end point
    :param point_of_interest: The node of interest ID
    :return: True if the point_of_interest is inside the nerve ring, or False of it isn't
    """
    parental_dict = {}
    for _index, _row in skeleton_tree.iterrows():
        if _row['type'] == 'root':
            parental_dict[_row['node_id']] = 'root'
        else:
            parental_dict[_row['node_id']] = _row['parent_id']
    current_point = point_of_interest
    in_nerve_ring = False
    while current_point != 'root':
        if current_point == nerve_ring_starts or current_point == nerve_ring_ends:
            in_nerve_ring = not in_nerve_ring
        current_point = parental_dict[current_point]
    return in_nerve_ring


class NeuronMorphology:
    """
    This class, stores the 3D morphological information of a neuron,
        including the positions of: skeleton, cell body, connectors (synapses)
    """
    def __init__(self, neuron_name: str, project_id: int):
        """
        :param neuron_name: Name of the neuron
        :param project_id: The project_id in which we are looking at the neuron

        Internal features:
        cell_body_center:   For a single neuron: the position of the cell body
        cell_body_center_L: For a neuronal class (L/R pair): the position of the left cell body
        cell_body_center_R: For a neuronal class (L/R pair): the position of the right cell body

        skeleton: List of skeleton points
        sending_connectors: List of positions of the sending synapses
        sending_connectors_labels: The label of sending synapses (i.e., sent to which post-synaptic neuron)
        receiving_connectors: List of positions of the receiving synapses
        receiving_connectors_labels: The label of receiving synapses (i.e., received from which pre-synaptic neuron)

        nerve_ring_starts: The id of a skeleton node, which identifies the nerve ring beginning
        nerve_ring_ends: The id of a skeleton node, which identifies the nerve ring ending
        """
        pymaid.CatmaidInstance(
            server='https://zhencatmaid.com/',
            api_token='c48243e19b85edf37345ced8049ce5d6c5802412',
            project_id=project_id
        )
        self.neuron_name = neuron_name
        self.project_id = project_id
        # #################################
        self.cell_body_center = None
        self.cell_body_center_L = None
        self.cell_body_center_R = None
        self.skeleton = []
        self.sending_connectors = []
        self.sending_connectors_labels = []
        self.receiving_connectors = []
        self.receiving_connectors_labels = []
        # ####################################
        self.nerve_ring_starts = None
        self.nerve_ring_ends = None

    def find_skeleton(self):
        """
        Filling the self.skeleton with skeleton nodes' positions
        :return: -
        """
        _neuron_info = pymaid.get_neuron(self.neuron_name).to_dataframe()
        """
        columns: ['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags']
        """
        # all_r = []
        # print(_neuron_info)
        # for _sub_neuron_skeleton in _neuron_info['nodes']:  # Mainly L & R
        for _ind, _sub_neuron in _neuron_info.iterrows():  # Mainly L & R
            _sub_neuron_tags = _sub_neuron['tags']
            """
            columns:
            ['ends', 'dendrite_starts', 'not a branch', 'TODO', 'nerve_ring_ends', 'nerve_ring_starts', 'nucleus']
            """
            _sub_neuron_name = refine_neuron_name(_sub_neuron['neuron_name'])
            # #################  Checking to ne inside the nerve ring #############
            if 'nerve_ring_ends' in _sub_neuron_tags.keys():
                _nerve_ring_starts_list = _sub_neuron_tags['nerve_ring_starts']
                _nerve_ring_ends_list = _sub_neuron_tags['nerve_ring_ends']
                if len(_nerve_ring_starts_list) != 1 or len(_nerve_ring_ends_list) != 1:
                    raise SyntaxError
                self.nerve_ring_starts = _nerve_ring_starts_list[0]
                self.nerve_ring_ends = _nerve_ring_ends_list[0]
            # ##################################################################
            _sub_neuron_skeleton = _sub_neuron['nodes']
            if _sub_neuron_name == self.neuron_name + '/R' or\
                    _sub_neuron_name == self.neuron_name + '/L' or \
                    len(_sub_neuron_skeleton) < 50:
                continue
            """
            columns: ['node_id', 'parent_id', 'creator_id', 'x', 'y', 'z', 'radius', 'confidence', 'type']
            """
            # type: ['slab', 'root', 'branch', 'end']
            # print(_sub_neuron_name, len(_sub_neuron_skeleton))
            for _, _skeleton_instance in _sub_neuron_skeleton.iterrows():
                # if _skeleton_instance['type'] == 'root':
                #     print(_neuron_info['neuron_name'])

                if self.nerve_ring_starts and self.nerve_ring_ends:
                    in_nerve_ring = is_inside_nerve_ring(
                        skeleton_tree=_sub_neuron_skeleton[['node_id', 'parent_id', 'type']],
                        nerve_ring_starts=self.nerve_ring_starts,
                        nerve_ring_ends=self.nerve_ring_ends,
                        point_of_interest=_skeleton_instance['node_id']
                    )
                else:
                    in_nerve_ring = True

                point_coord = (_skeleton_instance['x'], _skeleton_instance['y'], _skeleton_instance['z'])
                if in_nerve_ring:
                    self.skeleton.append(point_coord)
                # if _skeleton_instance['radius'] > 10:  # Cell body
                if _skeleton_instance['type'] == 'root':  # Cell body
                    # print(self.neuron_name, _skeleton_instance['radius'])
                    if self.cell_body_center:
                        if self.cell_body_center[0] > point_coord[0]:  # x1 (R) > x2 (L)
                            self.cell_body_center_R = self.cell_body_center
                            self.cell_body_center_L = point_coord
                        else:
                            self.cell_body_center_L = self.cell_body_center
                            self.cell_body_center_R = point_coord
                        self.cell_body_center = None
                    else:
                        self.cell_body_center = point_coord
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # _xs, _ys, _zs = points_to_coordinates(self.skeleton)
        # ax.scatter3D(_xs, _ys, _zs,
        #              c='r',  # to cnt
        #              s=all_r)
        # plt.show()

    def get_cell_body_center(self):
        """
        self.cell_body_center getter
        :return: -
        """
        if not self.cell_body_center:
            self.find_skeleton()
        return self.cell_body_center

    def get_skeleton(self):
        """
        self.skeleton getter
        :return:
        """
        self.find_skeleton()
        return self.skeleton

    def find_connectors(self):
        """
        This method, fills the connectors features:
            self.sending_connectors
            self.sending_connectors_labels
            self.receiving_connectors
            self.receiving_connectors_labels
        :return: -
        """
        _neuron_info = pymaid.get_neuron(self.neuron_name).to_dataframe()
        """
        columns: ['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags']
        """
        for _ind, _sub_neuron in _neuron_info.iterrows():  # Mainly L & R
            _sub_neuron_name = refine_neuron_name(_sub_neuron['neuron_name'])
            _sub_neuron_skeleton_len = len(_sub_neuron['nodes'])
            _sub_neuron_connectors = _sub_neuron['connectors']
            if _sub_neuron_name == self.neuron_name + '/R' or \
                    _sub_neuron_name == self.neuron_name + '/L' or \
                    _sub_neuron_skeleton_len < 50:
                continue
            _neuron_id = int(_sub_neuron['skeleton_id'])
            # print(_sub_neuron_connectors)
            # print(_sub_neuron_name, _sub_neuron['skeleton_id'])
            for _index, _row in _sub_neuron_connectors.iterrows():
                # ['node_id', 'connector_id', 'type' (0 or 1), 'x', 'y', 'z']  Todo: synapse vs. gap junc.
                _connector_id = int(_row['connector_id'])
                _connector_info = pymaid.get_connector_details(_connector_id)
                # ['connector_id', 'presynaptic_to', 'postsynaptic_to', 'presynaptic_to_node', 'postsynaptic_to_node']
                if len(_connector_info['presynaptic_to']) == 0:
                    continue
                connector_presynaptic_neuron = _connector_info['presynaptic_to'][0]
                connector_postsynaptic_neurons = _connector_info['postsynaptic_to'][0]
                if _neuron_id == connector_presynaptic_neuron:
                    _connector_coord = (_row['x'], _row['y'], _row['z'])
                    self.sending_connectors.append(_connector_coord)
                    postsynaptic_neurons_name = None
                    for post_neuron_id in connector_postsynaptic_neurons:
                        if postsynaptic_neurons_name:
                            postsynaptic_neurons_name += ',' + get_neuron_name_from_id(neuron_id=post_neuron_id)
                        else:
                            postsynaptic_neurons_name = get_neuron_name_from_id(neuron_id=post_neuron_id)
                    self.sending_connectors_labels.append(postsynaptic_neurons_name)
                elif _neuron_id in connector_postsynaptic_neurons:
                    _connector_coord = (_row['x'], _row['y'], _row['z'])
                    presynaptic_neuron_name = get_neuron_name_from_id(neuron_id=connector_presynaptic_neuron)
                    if presynaptic_neuron_name == 'large unknown fragment':
                        continue
                    self.receiving_connectors.append(_connector_coord)
                    self.receiving_connectors_labels.append(presynaptic_neuron_name)
                else:  # raise Exception
                    print("Connector is wrong")
                #     labels.append(str(pre_neuron_name) + "," + str(post_neuron_name))

    def get_connectors(self):
        """
        Connecots features getter (mind the order)
        :return: -
        """
        self.find_connectors()
        return self.sending_connectors, self.sending_connectors_labels, \
            self.receiving_connectors, self.receiving_connectors_labels


class PointTransformer:
    """
    This class, keeps an affine transformation, and can be applied on a point
    """
    def __init__(self,
                 scale_x: float, shift_x: float,
                 scale_y: float, shift_y: float,
                 scale_z: float, shift_z: float,
                 h_mirror: bool = False, v_mirror: bool = False,
                 diag_1: bool = False, diag_2: bool = False):
        """
        For an affine transformation, we have:
            x' = a1 x + b1
            y' = a2 y + b2
            z' = a3 z + b3
        :param scale_x: a1
        :param shift_x: b1
        :param scale_y: a2
        :param shift_y: b2
        :param scale_z: a3
        :param shift_z: b3
        :param h_mirror: Whether a horizontal flip is necessary before the affine transformation
        :param v_mirror: Whether a vertical flip is necessary before the affine transformation
        :param diag_1: Whether a diagonal flip is necessary before the affine transformation (\\)
        :param diag_2: Whether a diagonal flip is necessary before the affine transformation (//)
        """
        self.scale_x = scale_x
        self.shift_x = shift_x
        self.scale_y = scale_y
        self.shift_y = shift_y
        self.scale_z = scale_z
        self.shift_z = shift_z
        # #######################
        self.h_mirror = h_mirror
        self.v_mirror = v_mirror
        self.diag_1 = diag_1
        self.diag_2 = diag_2

    def transform_point(self, point):
        """
        Method to apply the transformation
        :param point: The point to be transformed
        :return: The transformed point
        """
        x_point, y_point, z_point = point[0], point[1], point[2]
        # ####### Pre process ########
        if self.h_mirror:
            x_point = - x_point
        if self.v_mirror:
            z_point = - z_point
        if self.diag_1:
            x_point, z_point = - z_point, - x_point
        if self.diag_2:
            x_point, z_point = z_point, x_point
        # ####### Affine transform #########
        x_transformed = x_point * self.scale_x + self.shift_x
        y_transformed = y_point * self.scale_y + self.shift_y
        z_transformed = z_point * self.scale_z + self.shift_z
        transformed_point = (x_transformed, y_transformed, z_transformed)
        return transformed_point


def find_alignment_transform(pin_points: list, base_points: list,
                             h_mirror: bool = False, v_mirror: bool = False,
                             diag_1: bool = False, diag_2: bool = False) -> PointTransformer:
    """
    This finction, find an affine transformation to fit pin_points to base_points
    :param pin_points: The points to be transformed (input points)
    :param base_points: Target of transformation (target points)
    :param h_mirror: Whether a horizontal flip is necessary before the affine transformation
    :param v_mirror: Whether a vertical flip is necessary before the affine transformation
    :param diag_1: Whether a diagonal flip is necessary before the affine transformation (\\)
    :param diag_2: Whether a diagonal flip is necessary before the affine transformation (//)
    :return: A PointTransformer object, defining the desired transformation between these set of points
    """
    pin_points_x, pin_points_y, pin_points_z = points_to_coordinates(pin_points)
    base_points_x, base_points_y, base_points_z = points_to_coordinates(base_points)
    # ####### Pre process ########
    if h_mirror:
        pin_points_x = [- x_point for x_point in pin_points_x]
    if v_mirror:
        pin_points_z = [- z_point for z_point in pin_points_z]
    if diag_1:
        pin_points_x, pin_points_z = [- z_point for z_point in pin_points_z], [- x_point for x_point in pin_points_x]
    if diag_2:
        pin_points_x, pin_points_z = pin_points_z, pin_points_x
    # ####### Affine transform #########
    x_model = LinearRegression()
    # x_model.fit(np.array(base_points_x).reshape(-1, 1), np.array(pin_points_x))
    x_model.fit(np.array(pin_points_x).reshape(-1, 1), np.array(base_points_x))
    y_model = LinearRegression()
    # y_model.fit(np.array(base_points_y).reshape(-1, 1), np.array(pin_points_y))
    y_model.fit(np.array(pin_points_y).reshape(-1, 1), np.array(base_points_y))
    z_model = LinearRegression()
    # z_model.fit(np.array(base_points_z).reshape(-1, 1), np.array(pin_points_z))
    z_model.fit(np.array(pin_points_z).reshape(-1, 1), np.array(base_points_z))
    scale_x, shift_x = x_model.coef_[0], x_model.intercept_
    scale_y, shift_y = y_model.coef_[0], y_model.intercept_
    scale_z, shift_z = z_model.coef_[0], z_model.intercept_
    return PointTransformer(scale_x=scale_x, shift_x=shift_x,
                            scale_y=scale_y, shift_y=shift_y,
                            scale_z=scale_z, shift_z=shift_z,
                            h_mirror=h_mirror, v_mirror=v_mirror,
                            diag_1=diag_1, diag_2=diag_2)


class CatmaidProject:
    """
    This class, keeps any desirable number of neurons in a specific project.
    Besides, one can align all the positional information with another project,
        using get_neuron_aligned_skeleton or get_neuron_aligned_connectors.
    """
    def __init__(self, project_id: int):
        self.project_id = project_id
        self.pin_neurons = [NeuronMorphology(neuron_name=pin_neuron, project_id=project_id)
                            for pin_neuron in PIN_NEURONS]
        # [pin_neuron.find_skeleton() for pin_neuron in self.pin_neurons]
        self.pin_points = [pin_neuron.get_cell_body_center() for pin_neuron in self.pin_neurons]
        print(self.pin_points)
        self.transform = PointTransformer(scale_x=1, shift_x=0,
                                          scale_y=1, shift_y=0,
                                          scale_z=1, shift_z=0)  # Default, base
        self.neurons = {}

    def get_pin_point(self) -> list:
        """
        self.pin_points getter
        :return: self.pin_points
        """
        return self.pin_points

    # def change_points(self, _points):
    #     _xs, _ys, _zs = points_to_coordinates(_points)
    #     new_points = []
    #     for _ind in range(len(_points)):
    #         # new_points.append((_zs[_ind], _ys[_ind], -_xs[_ind]))
    #         new_points.append((-_xs[_ind], _ys[_ind], -_zs[_ind]))
    #     return new_points

    def alignment_transform(self, base_pin_points: list, transformations: dict = None):
        """
        This method, find the transformation to align pin_points of this project to base_pin_points,
           and fills the self.transform
           This method also plots the base_pin_points and the self.pin_point
        :param base_pin_points: A list of (x, y, z)s for PIN_NEURONS, which we try to align self.pin_points on them
        :param transformations: (If applicable) extra non-affine transformations,
            including 'h_mirror' (horizontal flip), 'v_mirror' (vertical flip), 'diag_1' (diagonal flip, //),
            'diag_2' (diagonal flip, \\)
        :return: -
        """
        # ######################################
        if transformations is None:
            transformations = {'h_mirror': False, 'v_mirror': False,
                               'diag_1': False, 'diag_2': False}
        h_mirror = transformations['h_mirror']
        v_mirror = transformations['v_mirror']
        diag_1 = transformations['diag_1']
        diag_2 = transformations['diag_2']
        # #######################################
        self.transform = find_alignment_transform(pin_points=self.pin_points, base_points=base_pin_points,
                                                  h_mirror=h_mirror, v_mirror=v_mirror,
                                                  diag_1=diag_1, diag_2=diag_2)

        # new_points = [self.transform.transform_point(_point) for _point in self.pin_points]
        pin_labels = range(1, len(PIN_NEURONS) + 1)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        _xs, _ys, _zs = points_to_coordinates(self.pin_points)
        ax.scatter3D(_xs, _ys, _zs, c='r', s=10, label='Dataset1')
        for _ind, _label in enumerate(pin_labels):
            ax.text(_xs[_ind], _ys[_ind], _zs[_ind], '%s' % _label, size=5, zorder=1, color='k')

        _xs, _ys, _zs = points_to_coordinates(base_pin_points)
        ax.scatter3D(_xs, _ys, _zs, c='b', s=10, label='Dataset2')
        for _ind, _label in enumerate(pin_labels):
            ax.text(_xs[_ind], _ys[_ind], _zs[_ind], '%s' % _label, size=5, zorder=1, color='k')

        plt.title("Cell bodies (before alignment)")
        plt.legend(numpoints=1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def add_neuron(self, neuron_name: str):
        """
        Use this function to tell the class to keep the 3D information of a desired neuron, neuron_name
        :param neuron_name: The name of the neuron to be stored in the class
        :return: -
        """
        self.neurons[neuron_name] = NeuronMorphology(neuron_name=neuron_name, project_id=self.project_id)

    def get_neuron_aligned_skeleton(self, neuron_name: str):
        """
        This function, returns the aligned skeleton of neuron_name
        :param neuron_name: The name of the desired neuron
        :return: aligned_skeleton
        """
        non_aligned_skeleton = self.neurons[neuron_name].get_skeleton()
        [non_aligned_skeleton.append(_pin_point) for _pin_point in self.pin_points]  # Also, showing pin-points
        aligned_skeleton = [self.transform.transform_point(point=_point) for _point in non_aligned_skeleton]
        return aligned_skeleton

    def get_neuron_aligned_connectors(self, neuron_name: str):
        """
        This method, returns the aligned positioned of the neuron_name's connectors (synapses)
        :param neuron_name: The name of the desired neuron
        :return: aligned_sending_connectors (positions of sending synapses),
                 sending_labels (name of the post-synaptic synapses, matching the order of sending_connectors),
                 aligned_receiving_connectors (positions of receiving synapses),
                 receiving_labels (name of the pre-synaptic synapses, matching the order of receiving_connectors)
        """
        non_aligned_sending_connectors, sending_labels, non_aligned_receiving_connectors, receiving_labels = \
            self.neurons[neuron_name].get_connectors()
        aligned_sending_connectors = [self.transform.transform_point(point=_point)
                                      for _point in non_aligned_sending_connectors]
        aligned_receiving_connectors = [self.transform.transform_point(point=_point)
                                        for _point in non_aligned_receiving_connectors]
        return aligned_sending_connectors, sending_labels, aligned_receiving_connectors, receiving_labels


# Default 'h_mirror', 'v_mirror', 'diag_1', 'diag_2' transformations
PROJECTS_TRANSFORMS = {124: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': False},  # L1, 0h
                       161: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': False},  # L1, 5h
                       121: {'h_mirror': True, 'v_mirror': False, 'diag_1': False, 'diag_2': False},  # L1, 8h
                       98: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': False},  # L1, 16h
                       106: {'h_mirror': False, 'v_mirror': True, 'diag_1': False, 'diag_2': False},  # L2
                       125: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': False},  # L3
                       133: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': False},  # Adult TEM
                       135: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': True},  # Adult SEM
                       138: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': False},  # N2 dauer (RID)
                       156: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': False},  # N2 dauer
                       186: {'h_mirror': False, 'v_mirror': False, 'diag_1': True, 'diag_2': False},  # daf-2 dauer
                       266: {'h_mirror': False, 'v_mirror': False, 'diag_1': False, 'diag_2': False}}  # Todo: automatic


def get_labels_intersection(labels_1: list, labels_2: list):
    """
    This function, finds the similar and different neurons between labels_1 and labels_2
    :param labels_1: List of neurons on each synapse. Each element could be like 'n1,n2', means n1 and n2 neurons
                are the targets of a single synapse
    :param labels_2: Similar to labels_1
    :return: sharing_labels_1 (similar neurons between labels_1 and labels_2, but is a list with the similar length
                                to labels_1, where non-similar neurons are replaced with 'NULL'),
             exclusive_labels_1 (neurons in labels_1 but not in labels_2, which is a list with the similar length
                                to labels_1, where similar neurons are replaced with 'NULL'),
             sharing_labels_2 (similar neurons between labels_1 and labels_2, but is a list with the similar length
                                to labels_2, where non-similar neurons are replaced with 'NULL'),
             exclusive_labels_2 (neurons in labels_2 but not in labels_1, which is a list with the similar length
                                to labels_2, where similar neurons are replaced with 'NULL')
    """
    expanded_labels_1 = []
    for _label in labels_1:
        expanded_labels_1 += _label.split(',')
    expanded_labels_2 = []
    for _label in labels_2:
        expanded_labels_2 += _label.split(',')
    sharing_connections = []
    for _connection in expanded_labels_1:
        if _connection in expanded_labels_2:
            sharing_connections.append(_connection)
    # ###########################################
    sharing_labels_1 = []
    exclusive_labels_1 = []
    for _label in labels_1:
        _all_cons = _label.split(',')
        sharing_cons = []
        diff_cons = []
        for _con in _all_cons:
            if _con in sharing_connections:
                sharing_cons.append(_con)
            else:
                diff_cons.append(_con)
        if sharing_cons:
            sharing_labels_1.append(",".join(sharing_cons))
        else:
            sharing_labels_1.append('NULL')
        if diff_cons:
            exclusive_labels_1.append(",".join(diff_cons))
        else:
            exclusive_labels_1.append('NULL')

    sharing_labels_2 = []
    exclusive_labels_2 = []
    for _label in labels_2:
        _all_cons = _label.split(',')
        sharing_cons = []
        diff_cons = []
        for _con in _all_cons:
            if _con in sharing_connections:
                sharing_cons.append(_con)
            else:
                diff_cons.append(_con)
        if sharing_cons:
            sharing_labels_2.append(",".join(sharing_cons))
        else:
            sharing_labels_2.append('NULL')
        if diff_cons:
            exclusive_labels_2.append(",".join(diff_cons))
        else:
            exclusive_labels_2.append('NULL')

    return sharing_labels_1, exclusive_labels_1, sharing_labels_2, exclusive_labels_2


def filter_connections(connection_coordinates: list, sharing_labels: list, exclusive_labels: list):
    """
    This function, separates connection_coordinates to sharing_coordinates and exclusive_coordinates,
            based on sharing_labels and exclusive_labels
    :param connection_coordinates: The (x, y, z)s for all the connections
    :param sharing_labels: A list defining connection similar to another neuron
    :param exclusive_labels: A list defining connection different to another neuron
        All these lists have the same length, and irrelevant elements are replaced with 'NULL'
    :return: sharing_coordinates (coordinates only for the sharing connections),
             filtered_sharing_labels (labels only for the sharing connections),
             exclusive_coordinates (coordinates only for the exclusive connections),
             filtered_exclusive_labels (labels only for the exclusive connections)
    """
    sharing_coordinates = []
    exclusive_coordinates = []
    num_points = len(connection_coordinates)
    if num_points != len(sharing_labels) or num_points != len(exclusive_labels):
        raise SyntaxError
    for _ind in range(num_points):
        if sharing_labels[_ind] != 'NULL':
            sharing_coordinates.append(connection_coordinates[_ind])
        if exclusive_labels[_ind] != 'NULL':
            exclusive_coordinates.append(connection_coordinates[_ind])
    filtered_sharing_labels = [_item for _item in sharing_labels if _item != 'NULL']
    filtered_exclusive_labels = [_item for _item in exclusive_labels if _item != 'NULL']
    return sharing_coordinates, filtered_sharing_labels, exclusive_coordinates, filtered_exclusive_labels


def list_to_count_dict(list_of_items: list) -> dict:
    """
    This function, transforms a list to a dict, with elements counting as values
    :param list_of_items: A list with repetitive values
    :return: counts dict
    """
    counts = dict()
    for _item in list_of_items:
        counts[_item] = counts.get(_item, 0) + 1
    return counts


def plot_neuron(dataset_name_1, dataset_name_2, plot_title,
                neuron_skeleton_1, neuron_skeleton_2,
                sending_connectors_1, sending_labels_1,
                receiving_connectors_1, receiving_labels_1,
                sending_connectors_2, sending_labels_2,
                receiving_connectors_2, receiving_labels_2):
    """
    This function, plots a neuron of two different worms,
        and identifies all the sharing and exclusive connections
    :param dataset_name_1: Dataset 1 name (to be shown in the figure)
    :param dataset_name_2: Dataset 2 name (to be shown in the figure)
    :param plot_title: Title of the plot
    :param neuron_skeleton_1: List of (x, y, z)s for the first neuron's skeleton
    :param neuron_skeleton_2: List of (x, y, z)s for the second neuron's skeleton
    :param sending_connectors_1: List of (x, y, z)s for the first neuron's sending synapses
    :param sending_labels_1: List of labels for the first neuron's sending synapses
    :param receiving_connectors_1: List of (x, y, z)s for the first neuron's receiving synapses
    :param receiving_labels_1: List of labels for the first neuron's sending synapses
    :param sending_connectors_2: List of (x, y, z)s for the second neuron's sending synapses
    :param sending_labels_2: List of labels for the second neuron's sending synapses
    :param receiving_connectors_2: List of labels for the second neuron's sending synapses
    :param receiving_labels_2: List of labels for the first neuron's sending synapses
    :return: -
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    _xs1, _ys1, _zs1 = points_to_coordinates(neuron_skeleton_1)
    ax.scatter3D(_xs1, _ys1, _zs1, c='purple', s=3, label=dataset_name_1)
    # ax.plot3D(_xs1, _ys1, _zs1, c='purple', s=3, label=dataset_name_1)

    _xs1, _ys1, _zs1 = points_to_coordinates(sending_connectors_1)
    ax.scatter3D(_xs1, _ys1, _zs1, marker='>', c='green', s=40, label='out, ' + dataset_name_1)
    for _ind, _label in enumerate(sending_labels_1):
        ax.text(_xs1[_ind], _ys1[_ind], _zs1[_ind], '%s' % _label, size=5, zorder=1, color='green')

    _xs1, _ys1, _zs1 = points_to_coordinates(receiving_connectors_1)
    ax.scatter3D(_xs1, _ys1, _zs1, marker='<', c='red', s=40, label='in, ' + dataset_name_1)
    for _ind, _label in enumerate(receiving_labels_1):
        ax.text(_xs1[_ind], _ys1[_ind], _zs1[_ind], '%s' % _label, size=5, zorder=1, color='red')

    _xs2, _ys2, _zs2 = points_to_coordinates(neuron_skeleton_2)
    ax.scatter3D(_xs2, _ys2, _zs2, c='orange', s=3, label=dataset_name_2)
    # ax.plot3D(_xs2, _ys2, _zs2, c='orange', label=dataset_name_2)

    _xs2, _ys2, _zs2 = points_to_coordinates(sending_connectors_2)
    ax.scatter3D(_xs2, _ys2, _zs2, marker='>', c='blue', s=40, label='out, ' + dataset_name_2)
    for _ind, _label in enumerate(sending_labels_2):
        ax.text(_xs2[_ind], _ys2[_ind], _zs2[_ind], '%s' % _label, size=5, zorder=1, color='blue')

    _xs2, _ys2, _zs2 = points_to_coordinates(receiving_connectors_2)
    ax.scatter3D(_xs2, _ys2, _zs2, marker='<', c='black', s=40, label='in, ' + dataset_name_2)
    for _ind, _label in enumerate(receiving_labels_2):
        ax.text(_xs2[_ind], _ys2[_ind], _zs2[_ind], '%s' % _label, size=5, zorder=1, color='black')

    print(plot_title)
    print(list_to_count_dict(sending_labels_1))
    print(list_to_count_dict(sending_labels_2))
    print(list_to_count_dict(receiving_labels_1))
    print(list_to_count_dict(receiving_labels_2))
    plt.title(plot_title)
    plt.legend(numpoints=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


DATASET_NAMES = {124: 'L1(0h)', 161: 'L1(5h)', 121: 'L1(8h)', 98: 'L1(16h)', 125: 'L3', 186: 'daf-2'}


def draw_aligned_neuron(neuron_name: str,
                        base_project_id: int = 125,
                        other_project_id: int = 186):
    """
    [Main function]
    This function, handles aligning projects with ids base_project_id and other_project_id,
       and draws the neuron with name neuron_name with all the connections (synapses)
    :param neuron_name: The name of the neuron to be drawn
    :param base_project_id: The id for the base_project
    :param other_project_id: The id for the other project (the one that will be aligned)
    :return: -
    """
    # base_project_id Should be one of [124, 161, 98*, 125, 133]
    dataset_name_1 = DATASET_NAMES[base_project_id]
    dataset_name_2 = DATASET_NAMES[other_project_id]
    base_project = CatmaidProject(project_id=base_project_id)
    base_project.add_neuron(neuron_name=neuron_name)
    base_project_pin_points = base_project.get_pin_point()
    # #######################################################
    neuron_skeleton_1 = base_project.get_neuron_aligned_skeleton(neuron_name=neuron_name)
    # ################################################
    sending_connectors_1, sending_labels_1, receiving_connectors_1, receiving_labels_1 = \
        base_project.get_neuron_aligned_connectors(neuron_name=neuron_name)
    # ################################################
    # other_project_id -> Good: 186; *** Put the transforming one here ***
    other_project = CatmaidProject(project_id=other_project_id)
    other_project.alignment_transform(base_pin_points=base_project_pin_points,
                                      transformations=PROJECTS_TRANSFORMS[other_project_id])
    other_project.add_neuron(neuron_name=neuron_name)
    # ################################################
    neuron_skeleton_2 = other_project.get_neuron_aligned_skeleton(neuron_name=neuron_name)
    # ################################################
    sending_connectors_2, sending_labels_2, receiving_connectors_2, receiving_labels_2 =\
        other_project.get_neuron_aligned_connectors(neuron_name=neuron_name)
    # ################################################
    sending_shared_1, sending_new_1, sending_shared_2, sending_new_2 =\
        get_labels_intersection(labels_1=sending_labels_1, labels_2=sending_labels_2)
    receiving_shared_1, receiving_new_1, receiving_shared_2, receiving_new_2 = \
        get_labels_intersection(labels_1=receiving_labels_1, labels_2=receiving_labels_2)
    # ################################################
    send_sharing_coord_1, send_sharing_labels_1, send_exclusive_coord_1, send_exclusive_labels_1 = filter_connections(
        connection_coordinates=sending_connectors_1, sharing_labels=sending_shared_1, exclusive_labels=sending_new_1
    )
    send_sharing_coord_2, send_sharing_labels_2, send_exclusive_coord_2, send_exclusive_labels_2 = filter_connections(
        connection_coordinates=sending_connectors_2, sharing_labels=sending_shared_2, exclusive_labels=sending_new_2
    )
    rec_sharing_coord_1, rec_sharing_labels_1, rec_exclusive_coord_1, rec_exclusive_labels_1 = filter_connections(
        connection_coordinates=receiving_connectors_1, sharing_labels=receiving_shared_1, exclusive_labels=receiving_new_1
    )
    rec_sharing_coord_2, rec_sharing_labels_2, rec_exclusive_coord_2, rec_exclusive_labels_2 = filter_connections(
        connection_coordinates=receiving_connectors_2, sharing_labels=receiving_shared_2, exclusive_labels=receiving_new_2
    )
    # ################################################
    # plot_neuron(neuron_name=neuron_name, project_id_1=project_id_1, project_id_2=project_id_2,
    #             neuron_skeleton_1=neuron_skeleton_1, neuron_skeleton_2=neuron_skeleton_2,
    #             sending_connectors_1=sending_connectors_1, sending_labels_1=sending_labels_1,
    #             receiving_connectors_1=receiving_connectors_1, receiving_labels_1=receiving_labels_1,
    #             sending_connectors_2=sending_connectors_2, sending_labels_2=sending_labels_2,
    #             receiving_connectors_2=receiving_connectors_2, receiving_labels_2=receiving_labels_2)
    plot_neuron(dataset_name_1=dataset_name_1, dataset_name_2=dataset_name_2,
                plot_title=dataset_name_1 + " vs " + dataset_name_2 + ": " + str(neuron_name) + ', sharing synapses',
                neuron_skeleton_1=neuron_skeleton_1, neuron_skeleton_2=neuron_skeleton_2,
                sending_connectors_1=send_sharing_coord_1, sending_labels_1=send_sharing_labels_1,
                receiving_connectors_1=rec_sharing_coord_1, receiving_labels_1=rec_sharing_labels_1,
                sending_connectors_2=send_sharing_coord_2, sending_labels_2=send_sharing_labels_2,
                receiving_connectors_2=rec_sharing_coord_2, receiving_labels_2=rec_sharing_labels_2)
    plot_neuron(dataset_name_1=dataset_name_1, dataset_name_2=dataset_name_2,
                plot_title=dataset_name_1 + " vs " + dataset_name_2 + ": " + str(neuron_name) + ', varying synapses',
                neuron_skeleton_1=neuron_skeleton_1, neuron_skeleton_2=neuron_skeleton_2,
                sending_connectors_1=send_exclusive_coord_1, sending_labels_1=send_exclusive_labels_1,
                receiving_connectors_1=rec_exclusive_coord_1, receiving_labels_1=rec_exclusive_labels_1,
                sending_connectors_2=send_exclusive_coord_2, sending_labels_2=send_exclusive_labels_2,
                receiving_connectors_2=rec_exclusive_coord_2, receiving_labels_2=rec_exclusive_labels_2)
    # ################################################
