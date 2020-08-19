import tensorflow.keras as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dropout
from tcn import TCN


class Model(object):
    def __init__(self, type, input_shape, num_layers, num_neurons):
        self.type = type 
        self.input_shape = input_shape 
        self.num_layers = num_layers 
        self.num_neurons = num_neurons 

    def build(self, activation, optimizer, learning_rate, loss_fn):
        if self.type == 'lstm' or self.type == 'gru':
            if self.type == 'gru':
                nn = GRU
            else:
                nn = LSTM
            if self.num_layers >= 2:
                seq = True
            else:
                seq = False
            self.learning_rate = learning_rate

            # architecture
            self.model = Sequential()
            # input layer
            self.model.add(nn(self.num_neurons, return_sequences=seq,
                           input_shape=self.input_shape))
            # self.model.add(Dropout(0.2))
            # hidden layers
            for layer in range(1, self.num_layers):
                if layer == (self.num_layers - 1):
                    seq = False
                self.model.add(nn(self.num_neurons, activation='tanh',
                               kernel_initializer='glorot_uniform',
                               return_sequences=seq))
                self.model.add(Dropout(0.2))
            # output layer
            self.model.add(Dense(20, activation=activation, kernel_initializer='he_normal'))
            self.model.add(Dense(1))

        elif self.type == 'mlp':
            self.learning_rate = learning_rate

            self.model = Sequential()
            self.model.add(Input(shape=self.input_shape))
            self.model.add(Dense(128, activation=activation, kernel_initializer='he_normal'))
            self.model.add(Dense(64, activation=activation, kernel_initializer='he_normal'))
            self.model.add(Dense(1))

        elif self.type == 'tcn':
            self.learning_rate = learning_rate
            if self.num_layers >= 2:
                seq = True
            else:
                seq = False
            # architecture
            self.model = Sequential()
            # input layer
            self.model.add(TCN(nb_filters=self.num_neurons, kernel_size=2, nb_stacks=2, dilations=[1, 2, 4, 8],
                               return_sequences=seq, input_shape=self.input_shape))
            # hidden layers
            for layer in range(1, self.num_layers):
                if layer == (self.num_layers - 1):
                    seq = False
                self.model.add(TCN(nb_filters=self.num_neurons, kernel_size=2, nb_stacks=2, dilations=[1, 2, 4, 8],
                                   return_sequences=seq))
            # output layer
            self.model.add(Dense(20, activation=activation, kernel_initializer='he_normal'))
            self.model.add(Dense(1))

        # optimizer
        if optimizer == 'sgd':
            optim = K.optimizers.SGD(
                lr=learning_rate, momentum=0.9, nesterov=True
                )
        elif optimizer == 'rmsprop':
            optim = K.optimizers.RMSprop(
                learning_rate=learning_rate, rho=0.9
                )
        elif optimizer == 'adam':
            optim = K.optimizers.Adam(
                learning_rate=learning_rate
                )
        else: 
            raise ValueError('No such optimizer.')

        # loss function
        if loss_fn == 'mae':
            loss = 'mean_absolute_error'
        elif loss_fn == 'mse':
            loss = 'mean_squared_error'
        elif loss_fn == 'msle':
            loss = 'mean_squared_logarithmic_error'
        else:
            raise ValueError('No such loss function.')

        # build
        self.model.compile(optimizer=optim, loss=loss)
        print(f'\n{self.model.summary()}')
        # graphviz style model plot
        # K.utils.plot_model(
        #     self.model,
        #     to_file='figs/model.png',
        #     show_shapes=True,
        #     show_layer_names=True,
        #     rankdir="TB",
        #     expand_nested=True,
        # )

    def train(self, x_train, y_train, x_valid, y_valid, 
              epochs, batch_size, save_checkpoint=False, save_dir=None):
        def piecewise_constant_fn(epoch):
            if epoch < 50:
                return self.learning_rate
            return self.learning_rate * 0.1
        lr_scheduler = K.callbacks.LearningRateScheduler(piecewise_constant_fn)

        self.batch_size = batch_size

        if save_checkpoint:
            callback = K.callbacks.ModelCheckpoint(
                save_dir, monitor='val_loss', verbose=1,
                save_best_only=True, mode='min', period=1
                )
            losses = self.model.fit(
                x_train, y_train, epochs=epochs, batch_size=self.batch_size, 
                verbose=1, callbacks=[callback],
                validation_data=(x_valid, y_valid), shuffle=False
                )
        else:
            losses = self.model.fit(
                x_train, y_train, epochs=epochs, batch_size=self.batch_size, 
                verbose=1, callbacks=[],
                validation_data=(x_valid, y_valid), shuffle=False
                )
        return self.model, losses

    def evaluate(self, x_valid, y_valid, x_test, y_test):
        scores = self.model.evaluate(
            x_test, y_test, batch_size=self.batch_size, verbose=1
            )
        valid_scores = self.model.evaluate(
            x_valid, y_valid, batch_size=self.batch_size, verbose=1
            )
        return scores, valid_scores
