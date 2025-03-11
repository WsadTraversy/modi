import numpy as np
import keras as keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data():
    with open('data/danedynucz70.txt', 'r') as data:
        x_train = list()
        y_train = list()
        for _, line in enumerate(data):
            p = line.split()
            x_train.append(float(p[0]))
            y_train.append(float(p[1]))
    
    with open('data/danedynwer70.txt', 'r') as data:
        x_valid = list()
        y_valid = list()
        for _, line in enumerate(data):
            p = line.split()
            x_valid.append(float(p[0]))
            y_valid.append(float(p[1]))
    
    return np.array(x_valid), np.array(y_valid), np.array(x_train), np.array(y_train)

def normalization():
    x_valid, y_valid, x_train, y_train = load_data()

    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_x.fit(x_train.reshape(-1, 1))
    x_train = scaler_x.transform(x_train.reshape(-1, 1))
    x_valid = scaler_x.transform(x_valid.reshape(-1, 1))
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.reshape(-1, 1))
    y_train = scaler_y.transform(y_train.reshape(-1, 1))
    y_valid = scaler_y.transform(y_valid.reshape(-1, 1))

    data_train = np.array([x_train[2:-1], x_train[1:-2], x_train[:-3], y_train[2:-1],  y_train[1:-2], y_train[:-3]]).T
    data_train = data_train[0]
    data_valid = np.array([x_valid[2:-1], x_valid[1:-2], x_valid[:-3], y_valid[2:-1],  y_valid[1:-2], y_valid[:-3]]).T
    data_valid = data_valid[0]

    data_out_train = y_train[3:]
    data_out_valid = y_valid[3:]

    return data_train, data_out_train, data_valid, data_out_valid

def MAE(Y, Y_predicted):
    mae = 0
    for k in range(4, Y.size):
        mae = mae + np.abs(Y[k] - Y_predicted[k])
    mae = mae / Y.size
  
    return mae


def model(neurons):
    X_train, Y_train, X_valid, Y_valid = normalization()
    model = keras.Sequential()
    model.add(keras.Input(shape=(6, )))
    model.add(keras.layers.Dense(neurons, keras.activations.relu))
    model.add(keras.layers.Dense(1, activation="linear"))

    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.1,
        decay_steps=50,
        decay_rate=0.75
    )

    model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=[keras.metrics.MeanAbsoluteError]
    )

    clb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)
    
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32, callbacks=clb)
    
    accuracy_train = model.evaluate(X_train, Y_train)
    print(f"Accuracy train: {accuracy_train}%")

    accuracy_test = model.evaluate(X_valid, Y_valid)
    print(f"Accuracy valid: {accuracy_test}%")

    prediction_train = model.predict(X_train)
    prediction_valid = model.predict(X_valid)

    mae_train = MAE(Y_train, prediction_train)
    mae_test  = MAE(Y_valid, prediction_valid)
    print(mae_train, mae_test)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(prediction_train, c='blue', label='Zbiór uczący')
    plt.plot(Y_train, color='red', label='Model y(k)')
    plt.title('Model y(k) BEZ rekurencji na tle zbioru uczącego')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(prediction_valid, c='red', label='Zbiór weryfikujący')
    plt.plot(Y_valid, color='green', label='Model y(k)')
    plt.title('Model y(t) BEZ rekurencji na tle zbioru weryfikującego')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model(50)