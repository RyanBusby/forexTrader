import os
from datetime import datetime as dt
from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

class DataLoader():

    def __init__(self, currency_pair, configs):
        uname = os.getenv('uname')
        upass = os.getenv('upass')
        conn_str = 'postgresql://%s:%s@localhost/forex' % (uname, upass)
        self.engine = create_engine(conn_str)
        self.Base = automap_base()
        self.Base.prepare(self.engine, reflect=True)
        self.session = Session(self.engine)

        # specific relation names
        # if currency_pair[-1] == 'a':
        #     var_name = '%s_ASK' % currency_pair[:-1].upper()
        # else:
        #     var_name = '%s_BID' % currency_pair[:-1].upper()

        exec('self.table=self.Base.classes.%s' % currency_pair)

        earliest = dt.now() - timedelta(weeks=configs['data']['weeks'])

        selectable = self.session\
            .query(self.table)\
            .filter(self.table.bardate > earliest)\
            .order_by(self.table.bardate)\
            .statement

        self.df =\
        pd.read_sql(selectable, self.engine, index_col='bardate')

        # create mid column

        self.df['mid'] =\
        self.df.low + ((self.df.high - self.df.low) / 2)

        # group, or don't

        if configs['data']['resample']:
            rule = configs['data']['resample_rule']
            self.df = self.df.resample(rule).mean().dropna()

        split = int(len(self.df) * .85)

        self.data_train = \
        self.df.get(configs['data']['columns']).values[:split]
        self.data_test  = \
        self.df.get(configs['data']['columns']).values[split:]
        self.len_train = len(self.data_train)
        self.len_test  = len(self.data_test)
        self.len_train_windows = None
        self.ts = self.df.index[split:]

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        if normalise:
            data_windows = self.normalise_windows(
                    data_windows,
                    single_window=False
            )

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        if normalise:
            window =\
            self.normalise_windows(window, single_window=True)[0]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero
        Normalization is calculated per window, perhaps normalizing with sql
        alchemy would make more sense, normalize entire dataset'''
        normalised_data = []
        if single_window:
            window_data = [window_data]
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col =\
                    [
                        (
                            (
                                float(p) / float(window[0, col_i])
                            ) - 1
                        )\
                    for p in window[:, col_i]
                    ]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
