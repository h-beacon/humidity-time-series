from .utils import moving_average

import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
def figsize(scale, nplots=1):
    fig_width_pt = 390.0                               
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale 
    fig_height = fig_width*golden_mean*nplots
    fig_size = [fig_width, fig_height]
    return fig_size


def predict_plot(model, x_train, y_train, x_valid, y_valid, x_test, y_test, scaler, losses):
    # Function for inverse normalization
    def inverse_norm(model, x, y, scaler):
        pred = model.predict(x)
        pred = pred.reshape(-1, 1)
        y = y.reshape(-1, 1)
        x = x[:, 0, :]
        pred = np.concatenate((x, pred), axis=1)
        real = np.concatenate((x, y), axis=1)
        prediction = scaler.inverse_transform(pred)
        real_values = scaler.inverse_transform(real)
        prediction = prediction[:, -1]
        prediction = moving_average(prediction, periods=100)
        real_values = real_values[:, -1]
        return prediction, real_values
    # Validation inverse normalization
    train_prediction, train_real = inverse_norm(model, x_train, y_train, scaler)
    valid_prediction, valid_real = inverse_norm(model, x_valid, y_valid, scaler)
    test_prediction, test_real = inverse_norm(model, x_test, y_test, scaler)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, squeeze=True,
                            figsize=figsize(1.5, 1.2))
    axs[0, 0].plot(train_real, 'b-', label='True data')
    axs[0, 0].plot(train_prediction, 'r--', label='Predicted data')
    axs[0, 0].set_xlabel('Time point')
    axs[0, 0].set_ylabel('Humidity [%]')
    axs[0, 0].legend(loc='best')
    axs[0, 0].set_title('Training dataset')
    axs[0, 0].grid()

    axs[1, 0].plot(valid_real, 'b-', label='True data')
    axs[1, 0].plot(valid_prediction, 'r--', label='Predicted data')
    axs[1, 0].set_xlabel('Time point')
    axs[1, 0].set_ylabel('Humidity [%]')
    axs[1, 0].legend(loc='best')
    axs[1, 0].set_title('Validation dataset')
    axs[1, 0].grid()

    axs[0, 1].plot(test_real, 'b-', label='True data')
    axs[0, 1].plot(test_prediction, 'r--', label='Predicted data')
    axs[0, 1].set_xlabel('Time point')
    axs[0, 1].set_ylabel('Humidity [%]')
    axs[0, 1].legend(loc='best')
    axs[0, 1].set_title('Test dataset')
    axs[0, 1].grid()

    axs[1, 1].plot(losses.history['loss'], 'k-', label='Train')
    axs[1, 1].plot(losses.history['val_loss'], 'k--', label='Validation')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend(loc='best')
    axs[1, 1].set_title('Learning curves')
    axs[1, 1].grid()
 
    plt.tight_layout()
    plt.show()