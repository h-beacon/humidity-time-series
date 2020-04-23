import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from build_features import process_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', nargs='?', type=int, default=32)
parser.add_argument('--step', nargs='?', type=int, help='Timestep')
args = parser.parse_args()


save_dir = ".../github/Models/Model.5h" #PROMIJENITI ZA GITHUB


def evaluate(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1)
    print('\nModel loss on test set: \n Batch size: {}'.format(args.batch_size))
    print("{}: {}".format(model.metrics_names, scores))


def predict_plot(model, x_test, y_test, scaler):
    pred = model.predict(x_test)
    pred = pred.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    x_test = x_test[:, 0, :]
    prediction = np.concatenate((x_test, pred), axis=1)
    real = np.concatenate((x_test, y_test), axis=1)
    prediction = scaler.inverse_transform(prediction)
    real = scaler.inverse_transform(real)
    prediction = prediction[:, -1]
    real = real[:, -1]
    plt.plot(prediction, color='red', marker='o', markersize=1.8, linewidth=0.8, label='Prediction')
    plt.plot(real, color='green', marker='o', markersize=1.8, linewidth=0.8, label='Real value')
    plt.ylabel('Humidity', fontsize=14)
    plt.legend()
    plt.title('Model prediction')
    plt.show()


def load(path):
    model = keras.models.load_model(path)
    print('Loaded model with minimum loss')
    print(model.summary())
    return model


def main():
    x_train, y_train, x_test, y_test, scaler = process_data(data, args.step)
    model = load(save_dir)
    evaluate(model, x_test, y_test)
    predict_plot(model, x_test, y_test, scaler, losses=losses)


if __name__ == "__main__":
    main()