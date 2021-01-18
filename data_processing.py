import pandas as pd
import os


def split_train_test_df(input_df, ratio=0.6, drop_date_time=True):
    """ Splitting train and test dataset """
    training_df = input_df.loc[input_df.index[:int(len(input_df) * ratio)], :].reset_index(drop=True)
    testing_df = input_df.loc[input_df.index[int(len(input_df) * ratio):], :].reset_index(drop=True)
    print('lenght of training_df = {}; date_time from {} to {}'.format(len(training_df), 
                                                                       training_df.loc[0, "date_time"], 
                                                                       training_df.loc[training_df.index[-1], "date_time"]))
    print('lenght of testing_df = {}; date_time from {} to {}'.format(len(testing_df), 
                                                                      testing_df.loc[0, "date_time"], 
                                                                      testing_df.loc[testing_df.index[-1], "date_time"]))
    if drop_date_time:
        training_df.drop('date_time', axis=1, inplace=True)
        testing_df.drop('date_time', axis=1, inplace=True)
    return training_df, testing_df
    

def preprocess_data(input_df,           # result_df from FeaturesHelper.get_result(bool_dropna=True)
                    input_period=1):    # forward-looking period
    y_col = f'{input_period}m_fwd_chg'
    input_df.loc[:, y_col] = input_df['close'].pct_change(input_period).shift(-input_period) * 10000  # in bps
    feature_list = [i for i in input_df.columns if i not in ['date_time', 'open', 'high', 'low', 'close', 'volume', y_col]]
    res_df = input_df.loc[:, ['date_time', y_col] + feature_list].dropna().reset_index(drop=True)
    return res_df


def save_df(input_df, save_path):  # expect a df with columns ['y', 'x1', 'x2', 'x3' ...]
    if not os.path.exists(os.path.basename(save_path)):
        os.makedirs(os.path.basename(save_path))
    
    input_df.to_csv(save_path, index=False, header=False)
    print(f'Saved to {save_path}')
    return 