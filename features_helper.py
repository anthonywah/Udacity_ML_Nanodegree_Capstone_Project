import pandas as pd
import numpy as np
import timeit
from matplotlib import pyplot as plt


"""
This script is imported to help compute different indicators in notebook 3_target_features_processing
Parameters are fixed to common practice and not supposed to be changed for simplicity
Alternatives such as "pip install ta-lib" may be used also but self-defined functions will allow more flexibilities

Note: Some indicators are adjusted such that the mean of the indicators will be at 0
"""

class FeaturesHelper:
    def __init__(self, input_df):
        assert all([i in input_df.columns for i in ['date_time', 'open', 'high', 'low', 'close', 'volume']]) 
        self.res_df = input_df.copy()      # For returning a dataframe with all accumulated features
        self.computed_features = []        # Storing string name of all features computed
        self.tmp_df = self.res_df.copy()
        self.raw_df = self.res_df.copy()
    
    def reset(self):
        self.tmp_df = self.raw_df.copy()
        self.res_df = self.raw_df.copy()
        self.computed_features = []
        return
    
    def get_available_features(self):
        return [i[1:] for i in dir(self) if '__' not in i and i[0] == '_']
    
    def __get_x_to_y(self, x, y):  # in bps
        return ((self.tmp_df[x] / self.tmp_df[y]) - 1) * 10000
    
    def __xm_price_chg(self, x):  # in bps
        self.res_df.loc[:, f'{x}m_price_chg'] = self.tmp_df['close'].pct_change(x) * 10000
        self.computed_features.append(f'{x}m_price_chg')
        return
    
    def _macd(self):
        self.tmp_df.loc[:, 'MACD'] = self.tmp_df['close'].ewm(span=12).mean() - self.tmp_df['close'].ewm(span=26).mean()
        self.tmp_df.loc[:, 'Signal'] = self.tmp_df['MACD'].ewm(span=9).mean()
        self.tmp_df.loc[:, 'Divergence'] = self.tmp_df['MACD'] - self.tmp_df['Signal']
        self.res_df.loc[:, 'macd'] = self.tmp_df['Divergence']
        self.computed_features.append('macd')
        return
    
    def _rsi(self):  # not in days so we use 15m
        self.tmp_df.loc[:, 'chg'] = self.tmp_df['close'].pct_change()
        self.tmp_df.loc[:, '+chg'] = self.tmp_df['chg'].copy()
        self.tmp_df.loc[:, '-chg'] = self.tmp_df['chg'].copy()
        self.tmp_df.loc[self.tmp_df['+chg'] <= 0, '+chg'] = 0.0
        self.tmp_df.loc[self.tmp_df['-chg'] >= 0, '-chg'] = 0.0
        self.tmp_df.loc[:, 'avg_gain'] = self.tmp_df['+chg'].rolling(15).mean()
        self.tmp_df.loc[:, 'avg_loss'] = self.tmp_df['-chg'].rolling(15).mean()
        self.tmp_df.loc[:, 'avg_gain_loss'] = (self.tmp_df['avg_gain'] / (-self.tmp_df['avg_loss']))
        self.res_df.loc[:, 'rsi'] = 100 - (100 / (1 + self.tmp_df['avg_gain_loss'])) - 50  # Adjustment for 0 mean
        self.computed_features.append('rsi')
        return
    
    def _adx(self):
        period = 15
        self.tmp_df.loc[:, '+DM'] = self.tmp_df['high'].diff()
        self.tmp_df.loc[:, '-DM'] = -self.tmp_df['low'].diff()
        for i in ['+DM', '-DM']:
            self.tmp_df.loc[self.tmp_df[i].replace(np.nan, 0.0) < 0, i] = 0.0
        self.tmp_df.loc[:, '_tr1'] = self.tmp_df['high'] - self.tmp_df['low']
        self.tmp_df.loc[:, '_tr2'] = self.tmp_df['high'] - self.tmp_df['close'].shift()
        self.tmp_df.loc[:, '_tr3'] = self.tmp_df['low'] - self.tmp_df['close'].shift()
        self.tmp_df.loc[:, 'tr'] = self.tmp_df[['_tr1', '_tr2', '_tr3']].max(axis=1)
        for i in ['+DM', '-DM', 'tr']:
            self.tmp_df.loc[:, f'period_sum_{i}'] = self.tmp_df[i].rolling(period).sum()
            self.tmp_df.loc[:, f'smoothed_{i}'] = (self.tmp_df[f'period_sum_{i}'] - ((self.tmp_df[f'period_sum_{i}'].shift() / period) - self.tmp_df[i]).replace(np.nan, 0.0))
        self.tmp_df.loc[:, '+DI'] = 100 * self.tmp_df.loc[:, 'smoothed_+DM'] / self.tmp_df.loc[:, 'smoothed_tr']
        self.tmp_df.loc[:, '-DI'] = 100 * self.tmp_df.loc[:, 'smoothed_-DM'] / self.tmp_df.loc[:, 'smoothed_tr']
        self.tmp_df.loc[:, 'DX'] = 100 * (self.tmp_df['+DI'] - self.tmp_df['-DI']).abs() / (self.tmp_df['+DI'] + self.tmp_df['-DI']).abs()
        self.tmp_df.loc[:, '_DX'] = self.tmp_df['DX']
        self.tmp_df.loc[self.tmp_df.index[period:period * 2], '_DX'] = self.tmp_df.loc[self.tmp_df.index[period:period * 2], '_DX'].mean()
        self.tmp_df.loc[:, 'ADX'] = self.tmp_df['_DX'].ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        self.res_df.loc[:, 'adx'] = self.tmp_df['ADX'].copy()
        self.computed_features.append('adx')
        return
    
    def _obv(self):
        self.tmp_df.loc[:, 'chg'] = self.tmp_df['close'].pct_change()
        self.tmp_df.loc[:, 'obv_vol'] = self.tmp_df['volume'] * np.sign(self.tmp_df['chg'])
        self.res_df.loc[:, 'obv'] = self.tmp_df['obv_vol'].cumsum()
        self.computed_features.append('obv')
        return
    
    def _fso(self):  # not in days so we use 15m as the window
        self.tmp_df.loc[:, 'r_low'] = self.tmp_df['low'].rolling(15).min()
        self.tmp_df.loc[:, 'r_high'] = self.tmp_df['high'].rolling(15).max()
        self.tmp_df.loc[:, 'fso'] = (100 * (self.tmp_df['close'] - self.tmp_df['r_low']) / (self.tmp_df['r_high'] - self.tmp_df['r_low'])) - 50  # Adjustment for 0 mean
        self.res_df.loc[:, 'fso'] = self.tmp_df['fso'].copy()
        self.computed_features.append('fso')
        pass
    
    def _sso(self):
        if 'fso' not in self.tmp_df.columns:
            self.tmp_df.loc[:, 'r_low'] = self.tmp_df['low'].rolling(15).min()
            self.tmp_df.loc[:, 'r_high'] = self.tmp_df['high'].rolling(15).max()
            self.tmp_df.loc[:, 'fso'] = (100 * (self.tmp_df['close'] - self.tmp_df['r_low']) / (self.tmp_df['r_high'] - self.tmp_df['r_low'])) - 50
        self.res_df.loc[:, 'sso'] = self.tmp_df['fso'].rolling(3).mean()
        self.computed_features.append('sso')
        pass
    
    def _log_volume(self):
        self.res_df.loc[:, 'log_volume'] = np.log(self.tmp_df['volume'])
        self.computed_features.append('log_volume')
        return
    
    def _volume_chg(self):
        self.tmp_df.loc[:, 'VOLUME_CHG'] = self.tmp_df['volume'].pct_change()
        self.res_df.loc[:, 'volume_chg'] = self.tmp_df['VOLUME_CHG']
        self.computed_features.append('volume_chg')
        return
    
    def _5m_smoothed_volume_chg(self):
        if 'volume_chg' not in self.computed_features:
            self.tmp_df.loc[:, 'VOLUME_CHG'] = self.tmp_df['volume'].pct_change()
        self.res_df.loc[:, '5m_smoothed_volume_chg'] = self.tmp_df['VOLUME_CHG'].rolling(5).mean()
        self.computed_features.append('5m_smoothed_volume_chg')
        return
    
    def _close_to_open(self):
        self.res_df.loc[:, 'close_to_open'] = self.__get_x_to_y('close', 'open')
        self.computed_features.append('close_to_open')
        return
    
    def _close_to_high(self):
        self.res_df.loc[:, 'close_to_high'] = self.__get_x_to_y('close', 'high')
        self.computed_features.append('close_to_high')
        return

    def _close_to_low(self):
        self.res_df.loc[:, 'close_to_low'] = self.__get_x_to_y('close', 'low')
        self.computed_features.append('close_to_low')
        return
    
    def _1m_price_chg(self):
        self.__xm_price_chg(1)
        return
    
    def _15m_price_chg(self):
        self.__xm_price_chg(15)
        return
    
    def _60m_price_chg(self):
        self.__xm_price_chg(60)
        return
    
    def _15m_draw_down(self):  # in bps
        self.res_df.loc[:, '15m_draw_down'] = ((self.tmp_df['close'] / self.tmp_df['high'].rolling(15).max()) - 1) * 10000
        self.computed_features.append('15m_draw_down')
        return
    
    def _15m_draw_up(self):  # in bps
        self.res_df.loc[:, '15m_draw_up'] = ((self.tmp_df['close'] / self.tmp_df['low'].rolling(15).min()) - 1) * 10000
        self.computed_features.append('15m_draw_up')
        return
    
    def _15m_z_volume(self):
        roll = self.tmp_df['volume'].rolling(15)
        self.res_df.loc[:, '15m_z_volume'] = (self.tmp_df['volume'] - roll.mean()) / roll.std()
        self.computed_features.append('15m_z_volume')
    
    def _15m_z_price(self):
        roll = self.tmp_df['close'].rolling(15)
        self.res_df.loc[:, '15m_z_price'] = (self.tmp_df['close'] - roll.mean()) / roll.std()
        self.computed_features.append('15m_z_price')
        return
        
    def _15m_chg_std(self):
        self.tmp_df.loc[:, 'chg'] = self.tmp_df['close'].pct_change() 
        self.res_df.loc[:, '15m_chg_std'] = self.tmp_df['chg'].rolling(15).std()
        self.computed_features.append('15m_chg_std')
        return
    
    def _60m_draw_down(self):  # in bps
        self.res_df.loc[:, '60m_draw_down'] = ((self.tmp_df['close'] / self.tmp_df['high'].rolling(60).max()) - 1) * 10000
        self.computed_features.append('60m_draw_down')
        return

    def _60m_draw_up(self):  # in bps
        self.res_df.loc[:, '60m_draw_up'] = ((self.tmp_df['close'] / self.tmp_df['low'].rolling(60).min()) - 1) * 10000
        self.computed_features.append('60m_draw_up')
        return

    def _60m_z_volume(self):
        roll = self.tmp_df['volume'].rolling(60)
        self.res_df.loc[:, '60m_z_volume'] = (self.tmp_df['volume'] - roll.mean()) / roll.std()
        self.computed_features.append('60m_z_volume')

    def _60m_z_price(self):
        roll = self.tmp_df['close'].rolling(60)
        self.res_df.loc[:, '60m_z_price'] = (self.tmp_df['close'] - roll.mean()) / roll.std()
        self.computed_features.append('60m_z_price')
        return

    def _60m_chg_std(self):
        self.tmp_df.loc[:, 'chg'] = self.tmp_df['close'].pct_change()
        self.res_df.loc[:, '60m_chg_std'] = self.tmp_df['chg'].rolling(60).std()
        self.computed_features.append('60m_chg_std')
        return
    
    def run_features_list(self, input_features_list, log=True):
        all_start = timeit.default_timer()
        for one_feature in input_features_list:
            if one_feature in self.computed_features:
                print(f'{one_feature} exists already')
                continue
            start = timeit.default_timer()
            eval(f'self._{one_feature}()')
            if log:
                print(f'Computed {one_feature:<25} ; used {round(timeit.default_timer() - start, 3):<4}s')
        print(f'Finished {len(input_features_list)} features computation')
        return
    
    def get_result(self, bool_dropna=False):
        if bool_dropna:
            return_df = self.res_df.dropna().reset_index(drop=True)
        else:
            return_df = self.res_df.reset_index(drop=True)
        return return_df
    
    def get_computed_features(self):
        return self.computed_features.copy()
    
    def plot_feature(self, feature_list, n=1000):  # max 2 features in feature_list
        assert all([True if i in self.computed_features else False for i in feature_list])
        assert 1<= len(feature_list) <= 2
        sub_df = self.get_result(bool_dropna=True).iloc[range(n), :].reset_index(drop=True)
        fig = plt.figure(figsize=(24, 10))
        ax1, ax2 = plt.subplot(211), plt.subplot(212)

        ts = sub_df['date_time']
        vol = sub_df['volume']
        sub_df.loc[:, 'colors'] = 'green'
        sub_df.loc[sub_df['open'] - sub_df['close'] > 0, 'colors'] = 'red'

        ax1.plot(ts, sub_df['close'], color='black', label='Close price')
        ax1_sub = ax1.twinx()
        ax1_sub.bar(ts, vol, color=sub_df['colors'], width=1, align='center', alpha=0.3, label='Volume')
        ax1_sub.hlines(0, xmin=ts.tolist()[0], xmax=ts.tolist()[-1])
        
        ax2.plot(ts, sub_df[feature_list[0]], color='red', label=feature_list[0])
        
        if len(feature_list) == 2:
            ax2_sub = ax2.twinx()
            ax2_sub.plot(ts, sub_df[feature_list[1]], color='green', label=feature_list[1])
        
        ax1.set_title('BTCUSDT Close Prices')
        ax1.legend(loc=2)
        ax2.set_title(' | '.join(feature_list))
        ax2.legend(loc=2)
        if len(feature_list) == 2:
            ax2_sub.legend(loc=1)

        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax1.get_shared_x_axes().join(ax1, ax2)        
        plt.tight_layout()
        plt.show()
        return
     
        