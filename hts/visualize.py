from .utils import moving_average

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

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, squeeze=True)
    axs[0].plot(valid_real, 'b-', label='True data')
    axs[0].plot(valid_prediction, 'r-', label='Predicted data')
    axs[1].set_xlabel('Time point')
    axs[0].set_ylabel('Humidity %')
    axs[0].legend(loc='best')
    axs[0].set_title('Validation dataset')
    axs[0].grid()

    axs[1].plot(test_real, 'b-', label='True data')
    axs[1].plot(test_prediction, 'r-', label='Predicted data')
    axs[1].set_xlabel('Time point')
    axs[1].set_ylabel('Humidity %')
    axs[1].legend(loc='best')
    axs[1].set_title('Test dataset')
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    if losses:
        fig, ax = plt.subplots()
        ax.plot(losses.history['loss'], 'b-', label='Train')
        ax.plot(losses.history['val_loss'], 'r-', label='Validation')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.grid()
        plt.show()