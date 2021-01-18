import pandas as pd
import numpy as np
import os
import sys


def read_all_csvs():
    csv_file_list = [os.path.join('input_data', i) for i in sorted(os.listdir('input_data'))]
    df_list = []
    for ind, i in enumerate(csv_file_list):
        pct_ind = int((1 + ind) * 50 / len(csv_file_list))
        sys.stdout.write('\r')
        sys.stdout.write('Reading in progress: [{}{}] {}%'.format("=" * pct_ind, "-" * (50 - pct_ind), pct_ind * 100 / 50))
        df_list.append(pd.read_csv(i))
    df = pd.concat(df_list).reset_index(drop=True)
    df = df.loc[df['volume'] > 0, :].reset_index(drop=True)
    print('\nFinished reading {} csv files and {} entries found'.format(len(csv_file_list), df.shape[0]))
    return df
