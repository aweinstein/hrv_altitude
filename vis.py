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
    grid.map(plt.plot, 'f', 'Pxx')
    grid.set_axis_labels('frquency [Hz]', 'Pxx [s**2/Hz]')
    grid.add_legend()
    #grid.map(plt.fill_between, 'f', 'Pxx', alpha=0.2)

    for ax in grid.axes:
        ax.axvspan(VLF, LF, facecolor='b', alpha=0.2)
        ax.axvspan(LF, HF, facecolor='r', alpha=0.2)

# - Por que las amplitudes de algunos sujetos son tan altos?
#  metrics.loc[metrics['power_LFHF']>5][['subject', 'height', 'power_LFHF']]

#     subject  height  power_LFHF
# 8         5       1    9.716566
# 9         5       4   14.064073
# 15        8       4    5.110905
# 19       10       4    9.434847
# 20       11       1   17.123414
# 21       11       4   28.144739
# 22       12       1   11.633706
# 23       12       4   30.499976

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
