import glob
import os
import pandas as pd

import sp

prefixs = ('ZEPECG',   # ECG
           'ZEPGPAHR', # Heart Rate
           'ZEPGPAOK', # Data OK
           'ZEPRTRAD', # R to R (absolute, distinct)
           'ZEPGPARR', # Respiration Rate
           'ZEPGPAPO', # Posture
           'ZEPGPAAC', # Activity
           'ZEPGPABS', # Battery Status
           'ZEPGPAST', # Skin Temperature °C
       )

dir_data = 'raw_data'

dir_subject = ( '01HRV0410',
             '01HRV1640',
             '02HRV0410',
             '02HRV2640',
             '03HRV0510',
             '03HRV1640',
             '04HRV0240',
             '04HRV0510',
             '05HRV0510',
             '05HRV1040',
             '06HRV0510',
             '06HRV1640',
             '07HRV0240',
             '07HRV0510',
             '08HRV0240',
             '08HRV0510',
             '09HRV0610',
             '09HRV1640',
             '10HRV0610',
             '10HRV1040',
             '11HRV0610',
             '11HRV1040',
             '12HRV0510',
             '12HRV2640',
             # '13HRV0440ACC',
             # '13HRV0440ACC2',
             # '13HRV0440ACC3',
             # '13HRV0910',
             # '14HRV2310',
             '15HRV0540',
             '15HRV2310',
             # '16HRV2510',
)

def time_to_sec(ts):
    h, m, s  = [float(x) for x in ts.split(':')]
    return h * 3600 + m * 60 + s

def more_than_one_var():
    date = '2015-02-04-11-19-06'
    dfs = {}
    for prefix in prefixs:
        fn = '{}{}.csv'.format(prefix, date)
        print(fn)
        dfs[prefix[3:]] = pd.read_csv(fn, header=None, usecols=[1,2],
                                      parse_dates=True,
                                      converters={1:time_to_sec})

if __name__ == '__main__':
    dfs = []
    for dir_name in dir_subject:
        height = int(dir_name[-2])
        subject = int(dir_name[:2])
        path = os.path.join(dir_data, dir_name, 'ZEPRTRAD*.csv')
        fn_rr = glob.glob(path)[0]
        print('Parsing subject {:2} at height {}'.format(subject, height))
        df = pd.read_csv(fn_rr, header=None, usecols=[1,2],
                         names=['ts', 'RR'],
                         converters={1:time_to_sec})
        df['ts'] -= df.ts[0]
        df['subject'] = subject
        df['height'] = height
        df = df[['subject', 'height', 'RR']] # leave out timestamp for the
                                             # moment
        df['HR'] = sp.RR_to_HR(df['RR'])
        df['time'] = (df['RR'] / 1000).cumsum()

        dfs.append(df)

    hrv = pd.concat(dfs)
    hrv.to_pickle('hrv.pkl')
