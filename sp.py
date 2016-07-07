from itertools import product
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


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


def compute_metrics(hrv):
    subjects = hrv['subject'].unique()
    heights = hrv['height'].unique()
    SDNNs, RMSSDs, mean_RR, mean_HR = [], [], [], []
    hrv = hrv.set_index(['subject', 'height']).sort_index()
    for subject, height in product(subjects.tolist(), heights.tolist()):
        print('Calculating metrics for subject {} '
              'and height {}'.format(subject, height))
        RR = hrv.loc[(subject, height)]['RR'].as_matrix()
        HR = hrv.loc[(subject, height)]['HR'].as_matrix()
        SDNNs.append(np.std(RR, ddof=1))
        RMSSDs.append(np.sqrt(np.mean(np.diff(RR)**2)))
        mean_RR.append(np.mean(RR))
        mean_HR.append(np.mean(HR))

    index = pd.MultiIndex.from_product([subjects, heights],
                                       names=['subject', 'height'])
    df = pd.DataFrame(np.column_stack((mean_RR, mean_HR, SDNNs, RMSSDs)),
                      index=index,
                      columns=['meanRR', 'meanHR', 'SDNN', 'RMSSD'])
    return df

if __name__ == '__main__':
    hrv = pd.read_pickle('hrv.pkl') # hrv.pickle is generated by parse_data.py
    hrv_filtered = remove_outliers_all(hrv)
    hrv_5m = extract_segment(hrv_filtered)
    metrics = compute_metrics(hrv_5m)