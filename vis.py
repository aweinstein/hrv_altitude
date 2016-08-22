import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="ticks")

def plot_RR(hrv):
    grid = sns.FacetGrid(hrv, col='subject', hue='height', col_wrap=4,
                         legend_out=False)
    grid.map(plt.plot, 'time', 'RR')
    grid.set_axis_labels('time [s]', 'RR interval [ms]')
    grid.add_legend()

def plot_HR(hrv):
    grid = sns.FacetGrid(hrv, col='subject', hue='height', col_wrap=4,
                         legend_out=False)
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

def violin(metrics, var):
    # ax = sns.violinplot(x='height', y=var, data=metrics,
    #                     inner=None, color=".8")
    ax = sns.stripplot(x='height', y=var, data=metrics, jitter=0.05)
    left = ax.collections[0].get_offsets()
    right = ax.collections[1].get_offsets()
    x = np.vstack((right[:,0], left[:,0]))
    y = np.vstack((right[:,1], left[:,1]))
    plt.plot(x, y, lw=0.3, c='k')

    return ax


if __name__ == '__main__':
    metrics = pd.read_pickle('dfs/metrics.pkl')
    plt.close('all')

    # pair_plot(metrics.reset_index())
    ax = violin(metrics.reset_index(), 'power_LFHF')
    plt.show()
