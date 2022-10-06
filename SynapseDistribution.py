import pandas as pd
import matplotlib.pyplot as plt


class AdjacencyAnalyzer:
    def __init__(self, adjacency_matrix_filepath: str):
        self.adjacency_matrix = pd.read_csv(adjacency_matrix_filepath, index_col=[0])
        self.all_neurons = self.adjacency_matrix.columns.tolist()
        # self.input_synapses_distribution = None
        # self.output_synapses_distribution = None
        self.synapses_distribution = pd.DataFrame(columns=['Input', 'Output'], index=self.all_neurons)
        self.stacked = None
        self.find_synapses_distribution()
        # ###########################################################

    def find_synapses_distribution(self):
        # self.input_synapses_distribution = self.adjacency_matrix.sum(axis=1)  # Over columns
        # self.output_synapses_distribution = self.adjacency_matrix.sum(axis=0)  # Over rows

        # ############## Pairing #############
        self.stacked = self.adjacency_matrix.stack().reset_index(name='Value')
        self.stacked['pair'] = self.stacked['level_0'] + ',' + self.stacked['level_1']
        self.stacked.set_index('pair', inplace=True)
        self.stacked.drop(['level_0', 'level_1'], axis=1, inplace=True)
        # ####################################
        self.synapses_distribution['Input'] = self.adjacency_matrix.sum(axis=1)  # Over columns
        self.synapses_distribution['Output'] = self.adjacency_matrix.sum(axis=0)  # Over rows
        # pd.concat([self.input_synapses_distribution, self.output_synapses_distribution], axis=1).rename(
        #     columns={0: "Input", 1: "Output"})

    def get_mean_synapses(self):
        # return self.adjacency_matrix.replace(0, pd.np.NaN).stack().dropna().mean()

        mean_synapses = self.synapses_distribution['Input'].sum() / 224
        # mean_synapses = self.input_synapses_distribution.sum() / 224
        if mean_synapses != self.synapses_distribution['Output'].sum() / 224:
            raise ValueError("Bad matrix")
        mean_synapses = self.synapses_distribution[self.synapses_distribution["Input"] != 0]["Input"].mean()
        # mean_synapses = self.synapses_distribution[self.synapses_distribution["Output"] != 0]["Output"].mean()
        return mean_synapses * 224

    def plot_histograms(self):
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # fig.suptitle('Synapses distribution')

        # Input synapses
        # num_bins = int(self.input_synapses_distribution.max()) - int(self.input_synapses_distribution.min())
        max_synapse = self.synapses_distribution['Input'].max()
        normalized_synapse_send = self.synapses_distribution['Input'].div(max_synapse).round(2)
        # non_zeros_send = normalized_synapse_send[normalized_synapse_send != 0.0]
        # non_zeros.hist(bins=20)

        # max_contact = non_zeros.max()
        # normalized_non_zero_send = non_zeros.div(max_contact).round(2)

        # normalized_non_zero.sort_values().plot.bar(ax=ax1, subplots=True)
        # ax1.set_title("Send")

        # Output synapses
        # num_bins = int(self.output_synapses_distribution.max()) - int(self.output_synapses_distribution.min())
        max_synapse = self.synapses_distribution['Output'].max()
        normalized_synapse_receive = self.synapses_distribution['Output'].div(max_synapse).round(2)
        # non_zeros_receive = normalized_synapse_receive[normalized_synapse_receive != 0.0]

        # non_zeros = self.synapses_distribution['Output'][self.synapses_distribution['Output'] != 0.0]
        # normalized_non_zero_receive = non_zeros.div(max_contact).round(2)

        # non_zeros.hist(bins=20)

        max_synapse = self.stacked.max()
        normalized_pairs = self.stacked.div(max_synapse).round(2)
        return normalized_pairs  # Todo
        # return normalized_synapse_send, normalized_synapse_receive

        # normalized_non_zero.sort_values().plot.bar(ax=ax2, subplots=True)
        # ax2.set_title("Receive")
        # plt.show()

    def plot_in_and_out_synapses(self):
        x = self.synapses_distribution.sort_values(by=['Input'])['Input']
        y = self.synapses_distribution.sort_values(by=['Input'])['Output']

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
        x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

        # scatter points on the main axes
        main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
        plt.xlabel('Sending')

        # histogram on the attached axes
        x_hist.hist(x, 20, histtype='stepfilled',
                    orientation='vertical', color='gray')
        x_hist.invert_yaxis()
        plt.ylabel('Receiving')

        y_hist.hist(y, 20, histtype='stepfilled',
                    orientation='horizontal', color='gray')
        y_hist.invert_xaxis()

        plt.show()
