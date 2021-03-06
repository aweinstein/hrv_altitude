from collections import namedtuple
from itertools import product
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch

def RR_to_HR(RR):
    """Convert RR interval, in milliseconds, to heart rate."""
    return 1000 * 60 / RR


def remove_outliers(df, winlength=50, last=13, minbpm=50, maxbpm=120,
                    verbose=False):
    """Removes outliers from non interpolated heart rate.

    Based on https://github.com/milegroup/ghrv/blob/master/DataModel.py
    See also https://github.com/cran/RHRV/blob/master/R/FilterNIHR.R
    """
    # threshold initialization
    ulast = last
    umean = 1.5 * ulast
    HR = df['HR'].as_matrix()
    RR = df['RR'].as_matrix()
    beat_time = df['time'].as_matrix()
    index = 0
    removed = 0
    while index < len(HR):
        if index > 0:
            v = HR[max(index - winlength, 0):index]
            M = np.mean(v)
            cond_1 = 100 * abs((HR[index] - HR[index-1]) / HR[index-1]) < ulast
            cond_3 = 100 * abs((HR[index] - M) / M) < umean
        else:
            cond_1 = True
            cond_3 = True
        if index < len(HR) - 1:
            cond_2 = 100 * abs((HR[index] - HR[index+1]) / HR[index+1]) < ulast
        else:
            cond_2 = True

        cond_4 = (HR[index] > minbpm) and (HR[index] < maxbpm)
        if (cond_1 or cond_2 or cond_3) and cond_4:
            index += 1
        else: # Remove outlier
            HR = np.delete(HR, index)
            RR = np.delete(RR, index)
            beat_time = np.delete(beat_time, index)
            removed += 1

    if verbose:
        print('{} outliers removed'.format(removed))
    return beat_time, HR, RR


def remove_outliers_all(hrv, verbose=False):

    hrv = hrv.copy()
    subjects = hrv['subject'].unique()
    heights = hrv['height'].unique()
    hrv = hrv.set_index(['subject', 'height']).sort_index()
    dfs = []
    for subject, height in product(subjects.tolist(), heights.tolist()):
        if verbose:
            print('Removing outliers of subject '
                  '{} at height {}'.format(subject, height))
        bt, HR, RR = remove_outliers(hrv.loc[(subject, height)])
        df = pd.DataFrame({'RR':RR, 'HR':HR, 'time':bt})
        df['subject'] = subject
        df['height'] = height
        df = df[['subject', 'height', 'HR', 'RR', 'time']]
        dfs.append(df)

    hrv = pd.concat(dfs)
    return hrv


def extract_segment(hrv):
    """Extract a 5 minutes segment of HRV data.

    The segment is extracted from the middle of the time series.
    """
    hrv = hrv.copy()
    subjects = hrv['subject'].unique()
    heights = hrv['height'].unique()
    hrv = hrv.set_index(['subject', 'height']).sort_index()
    dfs = []
    for subject, height in product(subjects.tolist(), heights.tolist()):
        time = hrv.loc[(subject, height)]['time'].as_matrix()
        n = len(time)
        i = int(n / 2) - 1
        j = int(n / 2) + 1
        five_min = 60 * 5
        while (i > 0) and (j < (n - 1)):
            dt = time[j] - time[i]
            if dt >= five_min:
                break
            else:
                i -= 1
                j += 1
        df = hrv.loc[(subject, height)].iloc[i:j].reset_index()
        df['time'] -= df['time'][0]
        dfs.append(df)
    hrv_five_min = pd.concat(dfs)
    return hrv_five_min

def power_spectrum(time, RR):
    """Compute the power spectrum of the RR tachogram.

    Parameters
    ----------
    time : ndarray
        Time vector
    RR : ndarray
        RR intervals

    Notes
    -----
    https://rhenanbartels.wordpress.com/2014/04/06/first-post/
    """
    f_interp = interp1d(time, RR, 'cubic')
    fs = 1
    t = np.arange(0, 300, 1/fs)
    RR = f_interp(t)
    f, Pxx_den = welch(RR, fs, nperseg=256, detrend='linear')

    return f, Pxx_den

def compute_power_densities(hrv):
    subjects = hrv['subject'].unique()
    heights = hrv['height'].unique()
    dfs = []
    cols = ['subject', 'height', 'f', 'Pxx']
    hrv = hrv.set_index(['subject', 'height']).sort_index()
    for subject, height in product(subjects.tolist(), heights.tolist()):
        print('Calculating power spectrum of subject {:d} '
              'at height {:d}'.format(subject, height))
        RR = hrv.loc[(subject, height)]['RR'].as_matrix()
        time = hrv.loc[(subject, height)]['time'].as_matrix()
        f, Pxx_den = power_spectrum(time, RR / 1000)
        data = dict(subject=subject, height=height,
                    f=f, Pxx=Pxx_den)
        df = pd.DataFrame(data, columns=cols)
        dfs.append(df)

    spectrums = pd.concat(dfs).reset_index(drop=True)
    spectrums.to_pickle('dfs/spectrums.pkl')

FD = namedtuple('FD', ['peak_VLF', 'peak_LF', 'peak_HF',
                       'power_VLF', 'power_LF', 'power_HF', 'power_LFHF',
                        'pcpower_VLF', 'pcpower_LF', 'pcpower_HF',
                        'nupower_LF', 'nupower_HF'
                   ])

def fd_metrics(time, RR, correct=False):
    """Compute frequency domain metrics.

    Parameters
    ----------
    time : ndarray
        Time vector with the timestamp of the R wave.
    RR : ndarray
        Vector with the RR intervals. The unit of the intervals must be
    seconds.
    correct : bool
        If true, divide by mean value of RR before computing the
        spectrum. Default to False.

    Returns
    -------
    FD
        namedtuple with all the metrics.

    Notes
    -----
    See J. Sacha, "Why should one normalize heart rate variability with respect
    to average heart rate", Frontiers in Physiology
    """
    if correct:
        RR /= RR.mean()

    f, Pxx_den = power_spectrum(time, RR)

    VLF, LF, HF = 0.04, 0.15, 0.4

    f_VLF = f < VLF
    f_LF = (f >= VLF) & (f < LF)
    f_HF = (f >= LF) & (f < HF)

    P_VLF = Pxx_den[f_VLF]
    P_LF = Pxx_den[f_LF]
    P_HF = Pxx_den[f_HF]

    fmax_VLF = f[Pxx_den == P_VLF.max()][0]
    fmax_LF = f[Pxx_den == P_LF.max()][0]
    fmax_HF = f[Pxx_den == P_HF.max()][0]

    # Multiply by 1e6 to get results in ms2, as Kubios
    df = f[1] - f[0]
    Ptot_VLF = P_VLF.sum() * df * 1e6
    Ptot_LF = P_LF.sum() * df * 1e6
    Ptot_HF = P_HF.sum() * df * 1e6
    lfhf = Ptot_LF / Ptot_HF

    # in percent power
    total = Ptot_VLF + Ptot_LF + Ptot_HF
    pc_VLF = Ptot_VLF / total * 100
    pc_LF = Ptot_LF / total * 100
    pc_HF = Ptot_HF / total * 100

    # in normalized units
    total -= Ptot_VLF
    nu_LF = Ptot_LF / total * 100
    nu_HF = Ptot_HF / total * 100

    return FD(peak_VLF=fmax_VLF, peak_LF=fmax_LF, peak_HF=fmax_HF,
              power_VLF=Ptot_VLF, power_LF=Ptot_LF, power_HF=Ptot_HF,
              power_LFHF=lfhf,
              pcpower_VLF=pc_VLF, pcpower_LF=pc_LF, pcpower_HF=pc_HF,
              nupower_LF=nu_LF, nupower_HF=nu_HF)

def compute_metrics(hrv):
    subjects = hrv['subject'].unique()
    heights = hrv['height'].unique()
    SDNNs, RMSSDs, mean_RR, mean_HR = [], [], [], []
    peak_VLF, peak_LF, peak_HF = [], [], []
    power_VLF, power_LF, power_HF, power_LFHF = [], [], [], []
    power_VLF_norm, power_LF_norm, power_HF_norm = [], [], []
    pcpower_VLF, pcpower_LF, pcpower_HF = [], [], []
    nupower_LF, nupower_HF = [], []
    hrv = hrv.set_index(['subject', 'height']).sort_index()
    for subject, height in product(subjects.tolist(), heights.tolist()):
        print('Calculating metrics for subject {} '
              'and height {}'.format(subject, height))
        RR = hrv.loc[(subject, height)]['RR'].as_matrix()
        HR = hrv.loc[(subject, height)]['HR'].as_matrix()
        time = hrv.loc[(subject, height)]['time'].as_matrix()
        SDNNs.append(np.std(RR, ddof=1))
        RMSSDs.append(np.sqrt(np.mean(np.diff(RR)**2)))
        mean_RR.append(np.mean(RR))
        mean_HR.append(np.mean(HR))
        fd = fd_metrics(time, RR / 1000)
        peak_VLF.append(fd.peak_VLF)
        peak_LF.append(fd.peak_LF)
        peak_HF.append(fd.peak_HF)
        power_VLF.append(fd.power_VLF)
        power_LF.append(fd.power_LF)
        power_HF.append(fd.power_HF)
        power_LFHF.append(fd.power_LFHF)
        pcpower_VLF.append(fd.pcpower_VLF)
        pcpower_LF.append(fd.pcpower_LF)
        pcpower_HF.append(fd.pcpower_HF)
        nupower_LF.append(fd.nupower_LF)
        nupower_HF.append(fd.nupower_HF)
        fd = fd_metrics(time, RR / 1000, True)
        power_VLF_norm.append(fd.power_VLF)
        power_LF_norm.append(fd.power_LF)
        power_HF_norm.append(fd.power_HF)
    index = pd.MultiIndex.from_product([subjects, heights],
                                       names=['subject', 'height'])
    data = np.column_stack((mean_RR, mean_HR, SDNNs, RMSSDs,
                            peak_VLF, peak_LF, peak_HF,
                            power_VLF, power_LF, power_HF, power_LFHF,
                            pcpower_VLF, pcpower_LF, pcpower_HF,
                            nupower_LF, nupower_HF,
                            power_VLF_norm, power_LF_norm, power_HF_norm))
    cols = ['meanRR', 'meanHR', 'SDNN', 'RMSSD',
            'peak_VLF', 'peak_LF', 'peak_HF',
            'power_VLF', 'power_LF', 'power_HF', 'power_LFHF',
            'pcpower_VLF', 'pcpower_LF', 'pcpower_HF',
            'nupower_LF', 'nupower_HF',
            'power_VLF_norm', 'power_LF_norm', 'power_HF_norm']
    df = pd.DataFrame(data, index=index, columns=cols)
    return df

def run_compute_power_densities():
    hrv_5m = pd.read_pickle('dfs/hrv_5m.pkl')
    compute_power_densities(hrv_5m)

def run_compute_metrics():
    hrv = pd.read_pickle('dfs/hrv.pkl') # hrv.pickle is generated by parse_data.py
    hrv_filtered = remove_outliers_all(hrv)
    hrv_5m = extract_segment(hrv_filtered)
    metrics = compute_metrics(hrv_5m)
    metrics.to_pickle('dfs/metrics.pkl')
    metrics.to_excel('dfs/metrics.xlsx')
    return metrics

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Compare spectrum of subject 1 and 4
    spectrum = pd.read_pickle('dfs/spectrums.pkl')
    spectrum = spectrum.set_index(['subject', 'height']).sort_index()
    hrv_5m = pd.read_pickle('dfs/hrv_5m.pkl')
    hrv_5m = hrv_5m.set_index(['subject', 'height']).sort_index()

    # Data for subject 1
    f1 = spectrum.loc[(1,1)]['f'].as_matrix()
    P1 = spectrum.loc[(1,1)]['Pxx'].as_matrix()
    t1 = hrv_5m.loc[(1,1)]['time'].as_matrix()
    RR1 = hrv_5m.loc[(1,1)]['RR'].as_matrix()

    # Data for subject 4
    f4 = spectrum.loc[(4,1)]['f'].as_matrix()
    P4= spectrum.loc[(4,1)]['Pxx'].as_matrix()
    t4 = hrv_5m.loc[(4,1)]['time'].as_matrix()
    RR4 = hrv_5m.loc[(4,1)]['RR'].as_matrix()

    plt.close('all')
    _, axs = plt.subplots(2, 2)
    axs[0,0].plot(t1, RR1)
    axs[0,0].set(xlabel='time [s]', ylabel='RR [s]', title='subject 1')

    axs[0,1].plot(t4, RR4)
    axs[0,1].set(xlabel='time [s]', title='subject 4')

    axs[1,0].plot(f1, P1)
    axs[1,0].set(xlabel='f [Hz]', ylabel='P[s^2/Hz]')

    axs[1,1].plot(f4, P4)
    axs[1,1].set(xlabel='f [Hz]')

    plt.show()
