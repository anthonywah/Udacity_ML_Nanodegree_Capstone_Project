import time
import datetime
import sys
from binance.client import Client
from argparse import ArgumentParser


INPUT_DATA_DIR = './input_data'


class BinanceAPIHelper:
    def __init__(self):
        self.api_key = '<<< YOUR OWN BINANCE API KEY >>>'
        self.secret_key = '<<< YOUR OWN BINANCE SECRET KEY >>>'
        self.client = Client(self.api_key, self.secret_key)

    def get_historical_ohlc(self,
                            symbol='BTCUSDT',
                            start_date=None,
                            end_date=None,):
        if end_date is None:
            end_date = str(datetime.datetime.today().date())
        if start_date is None:
            start_date = str(datetime.datetime.today().date() - datetime.timedelta(days=180))
        interval = '1m'
        print(f'Start getting {interval} data from {start_date} to {end_date}')
        results = self.client.get_historical_klines(symbol=symbol,
                                                    interval=interval,
                                                    start_str=start_date,
                                                    end_str=end_date)
        res_dict = [{'date_time': str(datetime.datetime.fromtimestamp(i[0] / 1000.0)),
                     'open': float(i[1]), 'high': float(i[2]), 'low': float(i[3]), 'close': float(i[4]),
                     'volume': float(i[5])} for i in results]
        df = pd.DataFrame(res_dict)
        print(f'{symbol} {interval} data from {start_date} to {end_date} retrieved')
        df.loc[:, 'date'] = df.loc[:, 'date_time'].apply(lambda x: x[:10])
        df_gb = df.groupby('date')
        check_and_create_directory(os.path.join(HIST_KLINE_DATA_DIR, interval, symbol))
        for k, v in df_gb.groups.items():
            _df = df.loc[v, ['date_time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
            save_path = os.path.join(HIST_KLINE_DATA_DIR, f'{symbol}_{k}.csv')
            _df.to_csv(save_path, index=False)
            print(f'Saved at {os.path.basename(save_path)}')
        print(f'Saved {interval} data from {start_date} to {end_date}')
        return None


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--coin',       help='target_coin', action='store', desk='coin',       default='BTCUSDT')
    p.add_argument('-s', '--start_date', help='YYYY-MM-DD',  action='store', desk='start_date', default='2020-01-01')
    p.add_argument('-e', '--end_date',   help='YYYY-MM-DD',  action='store', desk='start_date', default='2020-12-31')
    args = p.parse_args(sys.argv[1:])

    helper = BinanceAPIHelper()
    helper.get_historical_ohlc(symbol=args.coin, start_date=args.start_date, end_date=args.end_date)
