# Todo: merge with the SynapseDistribution (identical functions)

import pandas as pd
import matplotlib.pyplot as plt


"""
'ADFL', 'ADFR', 'ADLL', 'ADLR', 'AFDL', 'AFDR', 'ALML', 'ALMR', 'ALNL', 'ALNR', 'AQR', 'ASEL', 'ASER', 'ASGL', 'ASGR', 
'ASHL', 'ASHR', 'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR', 'AUAL', 'AUAR', 'AVM', 'AWAL', 'AWAR', 'AWBL', 'AWBR',
'AWCL', 'AWCR', 'BAGL', 'BAGR', 'DVA', 'FLPL', 'FLPR', 'IL2DL', 'IL2DR', 'IL2L', 'IL2R', 'IL2VL', 'IL2VR', 
'OLLL', 'OLLR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR', 'PLNL', 'PLNR', 'SAADL', 'SAADR', 'SAAVL', 'SAAVR', 'SDQL', 'SDQR',
'URBL', 'URBR', 'URXL', 'URXR', 'URYDL', 'URYDR', 'URYVL', 'URYVR', 'ADAL', 'ADAR', 'AIAL', 'AIAR', 'AIBL', 'AIBR', 
'AINL', 'AINR', 'AIYL', 'AIYR', 'AIZL', 'AIZR', 'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER', 
'BDUL', 'BDUR', 'DVC', 'PVCL', 'PVCR', 'PVNL', 'PVNR', 'PVPL', 'PVPR', 'PVR', 'PVT', 'RIAL', 'RIAR', 'RIBL', 'RIBR',
'RIFL', 'RIFR', 'RIGL', 'RIGR', 'RIH', 'RIML', 'RIMR', 'RIPL', 'RIPR', 'RIR', 'IL1DL', 'IL1DR', 'IL1L', 'IL1R', 
'IL1VL', 'IL1VR', 'RIVL', 'RIVR', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'RMED', 'RMEL', 'RMER', 'RMEV',
'RMFL', 'RMFR', 'RMHL', 'RMHR', 'SIADL', 'SIADR', 'SIAVL', 'SIAVR', 'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR', 
'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR', 'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR', 'URADL', 'URADR', 'URAVL', 'URAVR', 
'ADEL', 'ADER', 'AIML', 'AIMR', 'ALA', 'AVFL', 'AVFR', 'AVHL', 'AVHR', 'AVJL', 'AVJR', 'AVKL', 'AVKR', 'AVL',
'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR', 'HSNL', 'HSNR', 'PVQL', 'PVQR', 'RICL', 'RICR', 'RID', 'RIS', 'RMGL', 'RMGR',
'BWM-DL01', 'BWM-DR01', 'BWM-VL01', 'BWM-VR01', 'BWM-DL02', 'BWM-DR02', 'BWM-VL02', 'BWM-VR02',
'BWM-DL03', 'BWM-DR03', 'BWM-VL03', 'BWM-VR03', 'BWM-DL04', 'BWM-DR04', 'BWM-VL04', 'BWM-VR04', 
'BWM-DL05', 'BWM-DR05', 'BWM-VL05', 'BWM-VR05', 'BWM-DL06', 'BWM-DR06', 'BWM-VL06', 'BWM-VR06', 
'BWM-DL07', 'BWM-DR07', 'BWM-VL07', 'BWM-VR07', 'BWM-DL08', 'BWM-DR08', 'BWM-VL08', 'BWM-VR08', 'CANL', 'CANR', \
'CEPshDL', 'CEPshDR', 'CEPshVL', 'CEPshVR', 'GLRDL', 'GLRDR', 'GLRL', 'GLRR', 'GLRVL', 'GLRVR', 'excgl'
"""
CLASSES = ['PVC', 'PLN', 'BWMD08D', 'AWC', 'BWM07V', 'AIZ', 'URAV', 'RID', 'SIAD', 'AWB', 'ALA', 'FLP', 'SMBV', 'AIN',
           'RIG', 'BWMD01D', 'SAAV', 'OLL', 'SIBV', 'RIR', 'RMEL/R', 'RMED', 'BWMD05D', 'ADE', 'GLRD', 'AVB', 'AQR',
           'ASH', 'SIAV', 'IL1V', 'URYV', 'CEPD', 'BWMD06D', 'ADF', 'ASE', 'RIM', 'PVP', 'CEPshD', 'DVA', 'URYD', 'RMF',
           'ALM', 'BDU', 'RIH', 'BWM05V', 'AIY', 'AVH', 'BWMD07D', 'AVL', 'SMBD', 'IL2V', 'URX', 'CAN', 'ASG', 'SAAD',
           'BWMD04D', 'SIBD', 'AIB', 'IL1L/R', 'GLRL+R', 'AVJ', 'IL2D', 'RMEV', 'RIF', 'CEPshV', 'HSN', 'GLRV', 'RMDD',
           'URB', 'IL2L/R', 'IL1D', 'RIC', 'ASK', 'CEPV', 'AIM', 'BWMD02D', 'AVF', 'DVC', 'RMDL/R', 'AVD', 'AUA',
           'OLQD', 'RMG', 'SMDD', 'ASI', 'RMH', 'AVE', 'BWM04V', 'AVK', 'AVA', 'AFD', 'ALN', 'RIV', 'BWMD03D', 'BWM06V',
           'ASJ', 'RIS', 'BAG', 'RMDV', 'URAD', 'SDQ', 'ADA', 'PVN', 'SMDV', 'ADL', 'PVQ', 'AWA', 'RIB', 'AIA', 'PVR',
           'RIP', 'RIA', 'BWM02V', 'AVM', 'PVT', 'OLQV', 'BWM03V', 'BWM01V', 'BWM08V']

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


class ContactAnalyzer:
    def __init__(self, contactome_filepath: str):
        self.contactome_aggregated_matrix = pd.read_csv(contactome_filepath, index_col=[0])
        self.all_neurons = self.contactome_aggregated_matrix.columns.tolist()
        self.contactome_matrices = []
        self.num_datasets = len(str(self.contactome_aggregated_matrix[self.all_neurons[0]].iloc[0]).split(','))
        self.separate_matrices()
        self.contacts_distributions = [pd.DataFrame(columns=['Area'], index=self.all_neurons)
                                       for _in in range(self.num_datasets)]
        # ###########################################################

    def separate_matrices(self):
        for _ind in range(self.num_datasets):
            self.contactome_matrices.append(self.contactome_aggregated_matrix.copy())
        for _neuron_1 in self.all_neurons:
            for _neuron_2 in self.all_neurons:
                contact_strings = self.contactome_aggregated_matrix[_neuron_2].loc[_neuron_1].split(',  ')
                for _ind in range(self.num_datasets):
                    self.contactome_matrices[_ind][_neuron_2].loc[_neuron_1] = float(contact_strings[_ind])
        # self.contactome_matrices.append(', , , , ')

    def find_contacts_distributions(self):
        for _ind in range(self.num_datasets):
            # Over columns, = over rows
            self.contacts_distributions[_ind]['Area'] = self.contactome_matrices[_ind].sum(axis=1)

    def make_class_matrix(self, matrix_index: int) -> pd.DataFrame:
        # class_neurons = list(set(LR_DICT.values()))
        class_matrix = pd.DataFrame(0.0, columns=CLASSES, index=CLASSES)
        for _neuron_1 in self.all_neurons:
            if _neuron_1 == 'excgl':
                continue
            _neuron_1_class = LR_DICT[_neuron_1]
            for _neuron_2 in self.all_neurons:
                if _neuron_2 == 'excgl':
                    continue
                _neuron_2_class = LR_DICT[_neuron_2]
                contact_value = self.contactome_matrices[matrix_index][_neuron_2].loc[_neuron_1]
                if _neuron_1_class == _neuron_2_class:  # IntraClass contact (R <-> L)
                    contact_value /= 2
                class_matrix[_neuron_2_class].loc[_neuron_1_class] = \
                    class_matrix[_neuron_2_class].loc[_neuron_1_class] + contact_value
        return class_matrix

    def get_mean_synapses(self) -> list:
        self.find_contacts_distributions()
        mean_contacts = []
        for _ind in range(self.num_datasets):
            # mean_synapses = self.contacts_distributions[_ind]['Area'].sum() / 224
            mean_contact = self.contacts_distributions[_ind][self.contacts_distributions[_ind]["Area"] != 0]["Area"].mean()
            mean_contacts.append(mean_contact)
        return mean_contacts

    def plot_histograms(self, matrix_index: int):
        class_matrix = self.make_class_matrix(matrix_index=matrix_index)
        # class_contacts_distributions = class_matrix.sum(axis=1)

        class_matrix = class_matrix.stack().reset_index(name='Value')
        class_matrix['pair'] = class_matrix['level_0'] + ',' + class_matrix['level_1']
        class_matrix.set_index('pair', inplace=True)
        class_matrix.drop(['level_0', 'level_1'], axis=1, inplace=True)
        class_contacts_distributions = class_matrix

        # fig, ax = plt.subplots()
        # fig.suptitle('Contact distribution')

        max_contact = class_contacts_distributions.max()
        normalized_class_contacts = class_contacts_distributions.div(max_contact).round(2)
        # normalized_non_zero = normalized_class_contacts[normalized_class_contacts != 0.0]
        return normalized_class_contacts

        # normalized_non_zero.sort_values().plot.bar(ax=ax, subplots=True)
        # ax.set_title("Net Surface Area")
        # plt.show()

