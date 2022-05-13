import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError


def get_rnn_forecaster(in_units: int, out_units: int, name: str = None, input_shape=None):
    opt = Adam(name='AdamOpt')
    loss = MeanAbsoluteError(name='MAE')
    model = Sequential(name=name)
    model.add(GRU(units=in_units,
                  name=f'RNN_1_relu', activation='relu', return_sequences=True))
    model.add(GRU(units=in_units, name='RNN_2_linear'))
    model.add(Dense(units=out_units,
                    activation='sigmoid',
                    name='OUT_sigmoid'))
    model.compile(optimizer=opt, loss=loss)
    if input_shape is not None:
        model.build([None, *input_shape[1:]])
    return model
