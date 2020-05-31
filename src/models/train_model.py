import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras
import matplotlib.pyplot as plt
from build_features import process_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--step', nargs='?', type=int, help='Timestep')
parser.add_argument('--layers', nargs='?', type=int, default=2)
parser.add_argument('--neurons1', nargs='?', type=int)
parser.add_argument('--neurons2', nargs='?', type=int)
parser.add_argument('--optimizer', nargs= '?', type=str, help='sgd, rmsprop or adam')
parser.add_argument('--loss', nargs='?', type=str, help='mse or msle')
parser.add_argument('--lr', nargs='?', type=float, default=0.001)
parser.add_argument('--epochs', nargs='+', type=int, help='List of epochs, can be 1 or more')
parser.add_argument('--batch_size', nargs='?', type=int, default=32)
parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard, default false')
args = parser.parse_args()


data = "" #PROMIJENITI ZA GITHUB
save_dir = ".../github/Models/Model.h5" #PROMIJENITI ZA GITHUB


def define_model(x_train):
    model = Sequential()
    if args.layers == 2:
        model.add(LSTM(args.neurons1, activation='relu', return_sequences=True,
                       input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(args.neurons2, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
    else:
        model.add(LSTM(args.neurons1, activation='relu',
                       input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
    if args.optimizer == 'sgd':
        optim = keras.optimizers.SGD(lr=args.lr, nesterov=True)
    elif args.optimizer == 'rmsprop':
        optim = keras.optimizers.RMSprop(learning_rate=args.lr, rho=0.9)
    elif args.optimizer == 'adam':
        optim = keras.optimizers.Adam(learning_rate=args.lr)
    if args.loss == 'msle':
        loss = 'mean_squared_logarithmic_error'
    else:
        loss = 'mean_squared_error'
    model.compile(optimizer=optim, loss=loss, metrics=['mae'])
    print(model.summary())
    return model


def train(model, save_dir, x_train, y_train, epoch):
    callback = keras.callbacks.ModelCheckpoint(save_dir, monitor='val_loss', verbose=1,
                                               save_best_only=True, mode='min', period=1)
    if args.tensorboard:
        logdir = "logs\\fit\\"  #PROMIJENITI ZA GITHUB
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
        callbacks = [callback, tensorboard_callback]
        losses = model.fit(x_train, y_train, epochs=epoch, batch_size=args.batch_size, verbose=1,
                           callbacks=callbacks, validation_split=0.2, shuffle=False)
    else:
        losses = model.fit(x_train, y_train, epochs=epoch, batch_size=args.batch_size, verbose=1,
                           callbacks=[callback], validation_split=0.2, shuffle=False)
    return model, losses


def plot_learning_curve(losses):
    plt.plot(losses.history['loss'], label='Train')
    plt.plot(losses.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()


def main():
    x_train, y_train, x_test, y_test, scaler = process_data(data, args.step)
    print('Train set shape: \n', x_train.shape, y_train.shape)
    print('Test set shape: \n', x_test.shape, y_test.shape)
    for epoch in args.epochs:
        model = define_model(x_train)
        epoch = int(epoch)
        model, losses = train(model, save_dir, x_train, y_train, epoch)
        print('\nStep: {} \nLearning rate: {} \nBatch size: {} \nEpochs: {}\n'.format(args.step, args.lr,
                                                                                      args.batch_size, epoch))
        plot_learning_curve(losses)


if __name__ == "__main__":
    main()
