import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
import keras
import matplotlib.pyplot as plt
import argparse
from Data_clean import clean


parser = argparse.ArgumentParser()
parser.add_argument('--rnn', nargs='?', type=str, help='lstm or gru')
parser.add_argument('--trainset', nargs='?', type=str, help='Deep or shallow sensor')
parser.add_argument('--step', nargs='?', type=int, help='Timestep')
parser.add_argument('--layers', nargs='?', type=int)
parser.add_argument('--neurons', nargs='?', type=int)
parser.add_argument('--optimizer', nargs='?', type=str, help='sgd, rmsprop or adam')
parser.add_argument('--loss', nargs='?', type=str, help='mse or mae')
parser.add_argument('--lr', nargs='?', type=float, default=0.01)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--batch_size', nargs='?', type=int)
parser.add_argument('--save', action='store_true', help='Save best model, default false')
args = parser.parse_args()


def split_sequences(sequences, n_steps):
    x, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def process_data(data, test_data, step):
    values = data.values
    test_values = test_data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    test_scaled = scaler.transform(test_values)
    split = int(len(data) * 0.9)
    test_split = int(len(test_data) * 0.9)
    data_train = scaled[:split]
    data_valid = scaled[split:]
    data_test = test_scaled[test_split:]
    x_train, y_train = split_sequences(data_train, step)
    x_valid, y_valid = split_sequences(data_valid, step)
    x_test, y_test = split_sequences(data_test, step)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, scaler


def piecewise_constant_fn(epoch):
    if epoch < 25:
        return args.lr
    else:
        return args.lr * 0.1


def define_model(x_train):
    model = Sequential()
    if args.rnn == 'gru':
        mod = GRU
    else:
        mod = LSTM
    if args.layers >= 2:
        seq = True
    else:
        seq = False
    model.add(mod(args.neurons, return_sequences=seq,
                  input_shape=(x_train.shape[1], x_train.shape[2])))
    # model.add(Dropout(0.2))
    if args.layers >= 2:
        for layer in range(1, args.layers):
            if layer == (args.layers - 1):
                seq = False
            model.add(mod(args.neurons, activation='elu', kernel_initializer='he_normal',
                          return_sequences=seq))
            # model.add(Dropout(0.2))
    model.add(Dense(1))
    if args.optimizer == 'sgd':
        optim = keras.optimizers.SGD(lr=args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'rmsprop':
        optim = keras.optimizers.RMSprop(learning_rate=args.lr, rho=0.9)
    elif args.optimizer == 'adam':
        optim = keras.optimizers.Adam(learning_rate=args.lr)
    if args.loss == 'mae':
        loss = 'mean_absolute_error'
    else:
        loss = 'mean_squared_error'
    model.compile(optimizer=optim, loss=loss, metrics=['mae'])
    print(model.summary())
    return model


def train(model, save_dir, x_train, y_train, x_valid, y_valid, epoch):
    lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
    if args.save:
        callback = keras.callbacks.ModelCheckpoint(save_dir, monitor='val_loss', verbose=1,
                                                   save_best_only=True, mode='min', period=1)
        losses = model.fit(x_train, y_train, epochs=epoch, batch_size=args.batch_size, verbose=1,
                           callbacks=[callback, lr_scheduler], validation_data=(x_valid, y_valid), shuffle=False)
    else:
        losses = model.fit(x_train, y_train, epochs=epoch, batch_size=args.batch_size, verbose=1,
                           callbacks=[lr_scheduler], validation_data=(x_valid, y_valid), shuffle=False)
    return model, losses


def evaluate(model, x_valid, y_valid, x_test, y_test):
    scores = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1)
    valid_scores = model.evaluate(x_valid, y_valid, batch_size=args.batch_size, verbose=1)
    print('\nModel loss on validation set: \n Batch size: {}'.format(args.batch_size))
    print("{}: {}".format(model.metrics_names, valid_scores))
    print('\nModel loss on test set: \n Batch size: {}'.format(args.batch_size))
    print("{}: {}".format(model.metrics_names, scores))


def predict_plot(model, x_valid, y_valid, x_test, y_test, scaler, losses=None):
    # Test inverse normalization
    test_pred = model.predict(x_test)
    test_pred = test_pred.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    x_test = x_test[:, 0, :]
    test_prediction = np.concatenate((x_test, test_pred), axis=1)
    test_real = np.concatenate((x_test, y_test), axis=1)
    prediction = scaler.inverse_transform(test_prediction)
    test_real = scaler.inverse_transform(test_real)
    test_prediction = prediction[:, -1]
    test_prediction = moving_average(test_prediction, periods=10)
    test_real = test_real[:, -1]
    # Validation inverse normalization
    valid_pred = model.predict(x_valid)
    valid_pred = valid_pred.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    x_valid = x_valid[:, 0, :]
    val_prediction = np.concatenate((x_valid, valid_pred), axis=1)
    valid_real = np.concatenate((x_valid, y_valid), axis=1)
    val_prediction = scaler.inverse_transform(val_prediction)
    valid_real = scaler.inverse_transform(valid_real)
    valid_prediction = val_prediction[:, -1]
    valid_prediction = moving_average(valid_prediction, periods=10)
    valid_real = valid_real[:, -1]
    if losses is not None:
        plt.plot(losses.history['loss'], label='Train')
        plt.plot(losses.history['val_loss'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.show()
    plt.subplot(1, 2, 1)
    plt.plot(valid_prediction, color='red', marker='o', markersize=1.8, linewidth=0.8, label='Prediction')
    plt.plot(valid_real, color='green', marker='o', markersize=1.8, linewidth=0.8, label='Real value')
    plt.ylabel('Humidity', fontsize=14)
    plt.legend()
    plt.title('Model prediction on validation set')
    plt.subplot(1, 2, 2)
    plt.plot(test_prediction, color='red', marker='o', markersize=1.8, linewidth=0.8, label='Prediction')
    plt.plot(test_real, color='green', marker='o', markersize=1.8, linewidth=0.8, label='Real value')
    plt.ylabel('Humidity', fontsize=14)
    plt.legend()
    plt.title('Model prediction on train set')
    plt.show()


def load(path):
    model = keras.models.load_model(path)
    print('Loaded model with minimum validation loss')
    return model


def main():
    if args.trainset == 'deep':
        data_path = ".../github/humidity-time-series/src/data/Senzor_zemlje_2_new.csv"    # TODO: namisti sebi pathove
        test_data_path = ".../github/humidity-time-series/src/data/Senzor_zemlje_new.csv"
        save_dir = ".../github/humidity-time-series/models/Model_deep.h5"
    elif args.trainset == 'shallow':
        data_path = ".../github/humidity-time-series/src/data/Senzor_zemlje_new.csv"
        test_data_path = ".../github/humidity-time-series/src/data/Senzor_zemlje_2_new.csv"
        save_dir = ".../github/humidity-time-series/models/Model_shallow.h5"
    csv = pd.read_csv(data_path)
    test_csv = pd.read_csv(test_data_path)
    data = clean(csv, args.step, temp=False, absolute=True)
    test_data = clean(test_csv, args.step, temp=False, absolute=True)
    x_train, y_train, x_valid, y_valid, x_test, y_test, scaler = process_data(data, test_data, args.step)
    print()
    print('Train set shape: \n', x_train.shape, y_train.shape)
    print('Valid set shape: \n', x_valid.shape, y_valid.shape)
    print('Test set shape: \n', x_test.shape, y_test.shape)
    for epoch in args.epochs:
        model = define_model(x_train)
        epoch = int(epoch)
        model, losses = train(model, save_dir, x_train, y_train, x_valid, y_valid, epoch)
        if args.save is True:
            model = load(save_dir)
        print(model.summary())
        print('\nStep: {} \nLearning rate: {} \nBatch size: {} \nEpochs: {}\n'.format(args.step, args.lr,
                                                                                      args.batch_size, epoch))
        """ Plotting trainset predictions instead of testset, just replace x_train and y_train """
        evaluate(model, x_valid, y_valid, x_train, y_train)
        predict_plot(model, x_valid, y_valid, x_train, y_train, scaler, losses=losses)


if __name__ == "__main__":
    main()
