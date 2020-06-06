from hts.utils import moving_average

import numpy as np 
import matplotlib.pyplot as plt

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

    plt.subplot(1, 2, 1)
    plt.plot(valid_prediction, color='red', marker='o', markersize=1.8, 
             linewidth=0.8, label='Prediction')
    plt.plot(valid_real, color='green', marker='o', markersize=1.8, 
             linewidth=0.8, label='Real value')
    plt.ylabel('Humidity')
    plt.legend(loc='best')
    plt.title('Model prediction on validation set')

    plt.subplot(1, 2, 2)
    plt.plot(test_prediction, color='red', marker='o', markersize=1.8, 
             linewidth=0.8, label='Prediction')
    plt.plot(test_real, color='green', marker='o', markersize=1.8, 
             linewidth=0.8, label='Real value')
    plt.ylabel('Humidity')
    plt.legend()
    plt.title('Model prediction on train set')

    plt.tight_layout()
    plt.show()

    if losses is not None:
        plt.plot(losses.history['loss'], label='Train')
        plt.plot(losses.history['val_loss'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.show()