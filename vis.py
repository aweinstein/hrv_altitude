import matplotlib.pyplot as plt
import seaborn as sns
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
