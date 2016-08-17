import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="ticks")

def plot_RR(hrv):
    grid = sns.FacetGrid(hrv, col='subject', hue='height', col_wrap=4, legend_out=False)
    grid.map(plt.plot, 'time', 'RR')
    grid.set_axis_labels('time [s]', 'RR interval [ms]')
    grid.add_legend()

def plot_HR(hrv):
    grid = sns.FacetGrid(hrv, col='subject', hue='height', col_wrap=4, legend_out=False)
    grid.map(plt.plot, 'time', 'HR')
    grid.set_axis_labels('time [s]', 'Heart rate [bpm]')
    grid.add_legend()

def pair_plot(metrics):

    cols = [#'meanRR',
            #'meanHR',
            'SDNN',
            'RMSSD',
            #'peak_VLF', 'peak_LF', 'peak_HF',
            #'power_VLF', 'power_LF', 'power_HF',
            'peak_HF',
            'power_LFHF',
            #'pcpower_VLF', 'pcpower_LF', 'pcpower_HF',
            #'nupower_LF', 'nupower_HF'
    ]
    sns.pairplot(metrics, hue='height', vars=cols)

if __name__ == '__main__':
    metrics = pd.read_pickle('dfs/metrics.pkl')
    plt.close('all')
    pair_plot(metrics.reset_index())
    plt.show()
