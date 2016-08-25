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

    return grid

def plot_HR(hrv):
    grid = sns.FacetGrid(hrv, col='subject', hue='height', col_wrap=4,
                         legend_out=False)
    grid.map(plt.plot, 'time', 'HR')
    grid.set_axis_labels('time [s]', 'Heart rate [bpm]')
    grid.add_legend()

    return grid

def plot_spectrum(sp):
    VLF, LF, HF = 0.04, 0.15, 0.4
    grid = sns.FacetGrid(sp, col='subject', hue='height', col_wrap=4,
                         legend_out=False)
    grid.map(plt.semilogy, 'f', 'Pxx')
    #grid.map(plt.plot, 'f', 'Pxx')
    grid.set_axis_labels('frquency [Hz]', 'P [s^2/Hz]')
    grid.add_legend()
    grid.set(xlim=(VLF, HF), ylim=(0,0.15))


    for ax in grid.axes:
        ax.axvspan(VLF, LF, facecolor='b', alpha=0.1)
        ax.axvspan(LF, HF, facecolor='r', alpha=0.1)

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


if __name__ == '__main__x':
    metrics = pd.read_pickle('dfs/metrics.pkl')
    plt.close('all')

    # pair_plot(metrics.reset_index())
    ax = violin(metrics.reset_index(), 'power_LFHF')
    plt.show()

if __name__ == '__main__':
    sp = pd.read_pickle('dfs/spectrums.pkl')
    plt.close('all')
    plot_spectrum(sp)
    plt.savefig('figures/spectrums.pdf')
    plt.show()
