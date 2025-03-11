import matplotlib.pyplot as plt
import numpy as np

def podpunkt_a():
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

    # Wykresy 
    # plt.plot(x_valid)
    # plt.title("Dane weryfikujące - wejście")
    # plt.xlabel("k")
    # plt.ylabel("u")
    # plt.show()
    
    return np.array(x_valid), np.array(y_valid), np.array(x_train), np.array(y_train)

def errors(y_train, y_valid, y_mod_train, y_mod_valid, y_mod_rek_train, y_mod_rek_valid):
    mse_train, mse_valid = float(), float()
    mse_train_rek, mse_valid_rek = float(), float()

    sse_train, sse_valid = float(), float()
    sse_train_rek, sse_valid_rek = float(), float()

    mse_train = np.mean((y_mod_train - y_train)**2)
    sse_train = np.sum((y_mod_train - y_train)**2)
    mse_train_rek = np.mean((y_mod_rek_train - y_train)**2)
    sse_train_rek = np.sum((y_mod_rek_train - y_train)**2)

    mse_valid = np.mean((y_mod_valid - y_valid)**2)
    sse_valid = np.sum((y_mod_valid - y_valid)**2)
    mse_valid_rek = np.mean((y_mod_rek_valid - y_valid)**2)
    sse_valid_rek = np.sum((y_mod_rek_valid - y_valid)**2)
    
    return mse_train, mse_valid, mse_train_rek, mse_valid_rek, sse_train, sse_valid, sse_train_rek, sse_valid_rek 

def plots(y_train, y_mod_train, y_valid, y_mod_valid, y_mod_rek_train, y_mod_rek_valid):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(y_train, c='blue', label='Zbiór uczący')
    plt.plot(y_mod_train, color='red', label='Model y(k)')
    plt.title('Model y(k) BEZ rekurencji na tle zbioru uczącego')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_valid, c='red', label='Zbiór weryfikujący')
    plt.plot(y_mod_valid, color='green', label='Model y(k)')
    plt.title('Model y(t) BEZ rekurencji na tle zbioru weryfikującego')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(y_train, c='blue', label='Zbiór uczący')
    plt.plot(y_mod_rek_train, color='red', label='Model y(k)')
    plt.title('Model y(k) z rekurencją na tle zbioru uczącego')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_valid, c='red', label='Zbiór weryfikujący')
    plt.plot(y_mod_rek_valid, color='green', label='Model y(k)')
    plt.title('Model y(k) z rekurencją na tle zbioru weryfikującego')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

def podpunkt_b(model_type):
    x_valid, y_valid, x_train, y_train = podpunkt_a()

    # Wyznacznie modelu
    if model_type == 11:
        kp = 1
        M = np.array([x_train[:-1], y_train[:-1]]).T
        y = y_train[1:]
    elif model_type == 21:
        kp = 2
        M = np.array([x_train[1:-1], x_train[:-2], y_train[1:-1], y_train[:-2]]).T
        y = y_train[2:]
    elif model_type == 31:
        kp = 3
        M = np.array([x_train[2:-1], x_train[1:-2], x_train[:-3], y_train[2:-1],  y_train[1:-2], y_train[:-3]]).T
        y = y_train[3:]
    else:
        raise ValueError("model_type must be equal: {11, 21, 31}")

    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(M.transpose(), M)), M.transpose()), y)
    
    y_mod_train, y_mod_rek_train = y_train[:kp], y_train[:kp]
    y_mod_valid, y_mod_rek_valid = y_valid[:kp], y_valid[:kp]
    for i in range(kp, len(x_train)):
        if model_type == 11:
            # bez rekurencji
            vector_train = [x_train[i-1], y_train[i-1]]
            vector_valid = [x_valid[i-1], y_valid[i-1]]

            y_mod_train = np.append(y_mod_train, (np.matmul(np.array(vector_train).T, w)))
            y_mod_valid = np.append(y_mod_valid, np.matmul(np.array(vector_valid).T, w))

            # rekurencja
            vector_train = [x_train[i-1], y_mod_rek_train[i-1]]
            vector_valid = [x_valid[i-1], y_mod_rek_valid[i-1]]

            y_mod_rek_train = np.append(y_mod_rek_train, np.matmul(np.array(vector_train).T, w))
            y_mod_rek_valid = np.append(y_mod_rek_valid, np.matmul(np.array(vector_valid).T, w))

        elif model_type == 21:
            # bez rekurencji
            vector_train = [x_train[i-1], x_train[i-2], y_train[i-1], y_train[i-2]]
            vector_valid = [x_valid[i-1], x_valid[i-2], y_valid[i-1], y_valid[i-2]]

            y_mod_train = np.append(y_mod_train, (np.matmul(np.array(vector_train).T, w)))
            y_mod_valid = np.append(y_mod_valid, np.matmul(np.array(vector_valid).T, w))

            # rekurencja
            vector_train = [x_train[i-1], x_train[i-2], y_mod_rek_train[i-1], y_mod_rek_train[i-2]]
            vector_valid = [x_valid[i-1], x_valid[i-2], y_mod_rek_valid[i-1], y_mod_rek_valid[i-2]]

            y_mod_rek_train = np.append(y_mod_rek_train, np.matmul(np.array(vector_train).T, w))
            y_mod_rek_valid = np.append(y_mod_rek_valid, np.matmul(np.array(vector_valid).T, w))

        elif model_type == 31:
            # bez rekurencji
            vector_train = [x_train[i-1], x_train[i-2], x_train[i-3], y_train[i-1], y_train[i-2], y_train[i-3]]
            vector_valid = [x_valid[i-1], x_valid[i-2], x_valid[i-3], y_valid[i-1], y_valid[i-2], y_valid[i-3]]

            y_mod_train = np.append(y_mod_train, (np.matmul(np.array(vector_train).T, w)))
            y_mod_valid = np.append(y_mod_valid, np.matmul(np.array(vector_valid).T, w))

            # rekurencja
            vector_train = [x_train[i-1], x_train[i-2], x_train[i-3], y_mod_rek_train[i-1], y_mod_rek_train[i-2], y_mod_rek_train[i-3]]
            vector_valid = [x_valid[i-1], x_valid[i-2], x_valid[i-3], y_mod_rek_valid[i-1], y_mod_rek_valid[i-2], y_mod_rek_valid[i-3]]

            y_mod_rek_train = np.append(y_mod_rek_train, np.matmul(np.array(vector_train).T, w))
            y_mod_rek_valid = np.append(y_mod_rek_valid, np.matmul(np.array(vector_valid).T, w))
    
    # Liczenie błędów
    mse_train, mse_valid, mse_train_rek, mse_valid_rek, sse_train, sse_valid, sse_train_rek, sse_valid_rek = errors(y_train, y_valid, y_mod_train, y_mod_valid, y_mod_rek_train, y_mod_rek_valid)
    print(f"Dane uczące: {mse_train:.4f}, {sse_train:.4f} \n Dane weryfikacyjne: {mse_valid:.4f}, {sse_valid:.4f} \n Dane uczące rekurencja: {mse_train_rek:.4f}, {sse_train_rek:.4f} \n Dane weryfikacyjne rekurencja: {mse_valid_rek:.4f}, {sse_valid_rek:.4f}")

    # Wykresy
    plots(y_train, y_mod_train, y_valid, y_mod_valid, y_mod_rek_train, y_mod_rek_valid)

def podpunkt_c(model_type, model_type_degree):
    x_valid, y_valid, x_train, y_train = podpunkt_a()
    x_matrix, y_matrix = list(), list()

    # Wyznacznie modelu
    if model_type == 1:
        kp = 1
        for i in range(model_type_degree):
            x_matrix.append(np.power(x_train[:-1], i+1))
            y_matrix.append(np.power(y_train[:-1], i+1))
        M = np.array(x_matrix+y_matrix).T
        y = y_train[1:]
    elif model_type == 2:
        kp = 2
        x_matrix1, y_matrix1 = list(), list()
        for i in range(model_type_degree):
            x_matrix.append(np.power(x_train[1:-1], i+1))
            y_matrix.append(np.power(y_train[1:-1], i+1))
            x_matrix1.append(np.power(x_train[:-2], i+1))
            y_matrix1.append(np.power(y_train[:-2], i+1))
        M = np.array(x_matrix+x_matrix1+y_matrix+y_matrix1).T
        
        y = y_train[2:]
    elif model_type == 3:
        kp = 3
        x_matrix1, x_matrix2, y_matrix1, y_matrix2 = list(), list(), list(), list()
        for i in range(model_type_degree):
            x_matrix.append(np.power(x_train[2:-1], i+1))
            y_matrix.append(np.power(y_train[2:-1], i+1))
            x_matrix1.append(np.power(x_train[1:-2], i+1))
            y_matrix1.append(np.power(y_train[1:-2], i+1))
            x_matrix2.append(np.power(x_train[:-3], i+1))
            y_matrix2.append(np.power(y_train[:-3], i+1))
        M = np.array(x_matrix+x_matrix1+x_matrix2+y_matrix+y_matrix1+y_matrix2).T

        y = y_train[3:]
    else:
        raise ValueError("model_type must be equal: {1, 2, 3}")
        
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(M.transpose(), M)), M.transpose()), y)

    y_mod_train, y_mod_rek_train = y_train[:kp], y_train[:kp]
    y_mod_valid, y_mod_rek_valid = y_valid[:kp], y_valid[:kp]
    x_matrix, x_matrix1, x_matrix2, y_matrix, y_matrix1, y_matrix2 = list(), list(), list(), list(), list(), list()
    for i in range(kp, len(x_train)):
        if model_type == 1:
            # bez rekurencji
            for j in range(model_type_degree):
                x_matrix.append(np.power(x_train[i-1], j+1))
                x_matrix1.append(np.power(x_valid[i-1], j+1))
                y_matrix.append(np.power(y_train[i-1], j+1))
                y_matrix1.append(np.power(y_valid[i-1], j+1))
            vector_train = x_matrix+y_matrix
            x_matrix, y_matrix = list(), list()

            vector_valid = x_matrix1+y_matrix1
            x_matrix1, y_matrix1 = list(), list()

            y_mod_train = np.append(y_mod_train, (np.matmul(np.array(vector_train).T, w)))
            y_mod_valid = np.append(y_mod_valid, np.matmul(np.array(vector_valid).T, w))

            # rekurencja
            for j in range(model_type_degree):
                x_matrix.append(np.power(x_train[i-1], j+1))
                x_matrix1.append(np.power(x_valid[i-1], j+1))
                y_matrix.append(np.power(y_mod_rek_train[i-1], j+1))
                y_matrix1.append(np.power(y_mod_rek_valid[i-1], j+1))
            vector_train = x_matrix+y_matrix
            x_matrix, y_matrix = list(), list()

            vector_valid = x_matrix1+y_matrix1
            x_matrix1, y_matrix1 = list(), list()

            y_mod_rek_train = np.append(y_mod_rek_train, np.matmul(np.array(vector_train).T, w))
            y_mod_rek_valid = np.append(y_mod_rek_valid, np.matmul(np.array(vector_valid).T, w))
        
        elif model_type == 2:
            # bez rekurencji
            for j in range(model_type_degree):
                x_matrix.append(np.power(x_train[i-1], j+1))
                x_matrix1.append(np.power(x_train[i-2], j+1))
                y_matrix.append(np.power(y_train[i-1], j+1))
                y_matrix1.append(np.power(y_train[i-2], j+1))
            vector_train = x_matrix+x_matrix1+y_matrix+y_matrix1
            x_matrix, y_matrix, x_matrix1, y_matrix1 = list(), list(), list(), list()

            for j in range(model_type_degree):
                x_matrix.append(np.power(x_valid[i-1], j+1))
                x_matrix1.append(np.power(x_valid[i-2], j+1))
                y_matrix.append(np.power(y_valid[i-1], j+1))
                y_matrix1.append(np.power(y_valid[i-2], j+1))
            vector_valid = x_matrix+x_matrix1+y_matrix+y_matrix1
            x_matrix, y_matrix, x_matrix1, y_matrix1 = list(), list(), list(), list()

            y_mod_train = np.append(y_mod_train, (np.matmul(np.array(vector_train).T, w)))
            y_mod_valid = np.append(y_mod_valid, np.matmul(np.array(vector_valid).T, w))

            # rekurencja
            for j in range(model_type_degree):
                x_matrix.append(np.power(x_train[i-1], j+1))
                x_matrix1.append(np.power(x_train[i-2], j+1))
                y_matrix.append(np.power(y_mod_rek_train[i-1], j+1))
                y_matrix1.append(np.power(y_mod_rek_train[i-2], j+1))
            vector_train = x_matrix+x_matrix1+y_matrix+y_matrix1
            x_matrix, y_matrix, x_matrix1, y_matrix1 = list(), list(), list(), list()

            for j in range(model_type_degree):
                x_matrix.append(np.power(x_valid[i-1], j+1))
                x_matrix1.append(np.power(x_valid[i-2], j+1))
                y_matrix.append(np.power(y_mod_rek_valid[i-1], j+1))
                y_matrix1.append(np.power(y_mod_rek_valid[i-2], j+1))
            vector_valid = x_matrix+x_matrix1+y_matrix+y_matrix1
            x_matrix, y_matrix, x_matrix1, y_matrix1 = list(), list(), list(), list()

            y_mod_rek_train = np.append(y_mod_rek_train, np.matmul(np.array(vector_train).T, w))
            y_mod_rek_valid = np.append(y_mod_rek_valid, np.matmul(np.array(vector_valid).T, w))

        elif model_type == 3:
            # bez rekurencji
            for j in range(model_type_degree):
                x_matrix.append(np.power(x_train[i-1], j+1))
                x_matrix1.append(np.power(x_train[i-2], j+1))
                x_matrix2.append(np.power(x_train[i-3], j+1))
                y_matrix.append(np.power(y_train[i-1], j+1))
                y_matrix1.append(np.power(y_train[i-2], j+1))
                y_matrix2.append(np.power(y_train[i-3], j+1))
            vector_train = x_matrix+x_matrix1+x_matrix2+y_matrix+y_matrix1+y_matrix2
            x_matrix, y_matrix, x_matrix1, y_matrix1, x_matrix2, y_matrix2 = list(), list(), list(), list(), list(), list()

            for j in range(model_type_degree):
                x_matrix.append(np.power(x_valid[i-1], j+1))
                x_matrix1.append(np.power(x_valid[i-2], j+1))
                x_matrix2.append(np.power(x_valid[i-3], j+1))
                y_matrix.append(np.power(y_valid[i-1], j+1))
                y_matrix1.append(np.power(y_valid[i-2], j+1))
                y_matrix2.append(np.power(y_valid[i-3], j+1))
            vector_valid = x_matrix+x_matrix1+x_matrix2+y_matrix+y_matrix1+y_matrix2
            x_matrix, y_matrix, x_matrix1, y_matrix1, x_matrix2, y_matrix2 = list(), list(), list(), list(), list(), list()

            y_mod_train = np.append(y_mod_train, (np.matmul(np.array(vector_train).T, w)))
            y_mod_valid = np.append(y_mod_valid, np.matmul(np.array(vector_valid).T, w))

            # rekurencja
            for j in range(model_type_degree):
                x_matrix.append(np.power(x_train[i-1], j+1))
                x_matrix1.append(np.power(x_train[i-2], j+1))
                x_matrix2.append(np.power(x_train[i-3], j+1))
                y_matrix.append(np.power(y_mod_rek_train[i-1], j+1))
                y_matrix1.append(np.power(y_mod_rek_train[i-2], j+1))
                y_matrix2.append(np.power(y_mod_rek_train[i-3], j+1))
            vector_train = x_matrix+x_matrix1+x_matrix2+y_matrix+y_matrix1+y_matrix2
            x_matrix, y_matrix, x_matrix1, y_matrix1, x_matrix2, y_matrix2 = list(), list(), list(), list(), list(), list()

            for j in range(model_type_degree):
                x_matrix.append(np.power(x_valid[i-1], j+1))
                x_matrix1.append(np.power(x_valid[i-2], j+1))
                x_matrix2.append(np.power(x_valid[i-3], j+1))
                y_matrix.append(np.power(y_mod_rek_valid[i-1], j+1))
                y_matrix1.append(np.power(y_mod_rek_valid[i-2], j+1))
                y_matrix2.append(np.power(y_mod_rek_valid[i-3], j+1))
            vector_valid = x_matrix+x_matrix1+x_matrix2+y_matrix+y_matrix1+y_matrix2
            x_matrix, y_matrix, x_matrix1, y_matrix1, x_matrix2, y_matrix2 = list(), list(), list(), list(), list(), list()

            y_mod_rek_train = np.append(y_mod_rek_train, np.matmul(np.array(vector_train).T, w))
            y_mod_rek_valid = np.append(y_mod_rek_valid, np.matmul(np.array(vector_valid).T, w))
            
    # Liczenie błędów
    mse_train, mse_valid, mse_train_rek, mse_valid_rek, sse_train, sse_valid, sse_train_rek, sse_valid_rek = errors(y_train, y_valid, y_mod_train, y_mod_valid, y_mod_rek_train, y_mod_rek_valid)
    print(f"Dane uczące: {mse_train:.4f}, {sse_train:.4f} \n Dane weryfikacyjne: {mse_valid:.4f}, {sse_valid:.4f} \n Dane uczące rekurencja: {mse_train_rek:.4f}, {sse_train_rek:.4f} \n Dane weryfikacyjne rekurencja: {mse_valid_rek:.4f}, {sse_valid_rek:.4f}")

    # Wykresy
    plots(y_train, y_mod_train, y_valid, y_mod_valid, y_mod_rek_train, y_mod_rek_valid)

    # Podpunkt d
    # plt.scatter(x_valid, y_mod_rek_valid)
    # plt.title("Charakterystyka statyczna modelu nieliniowego")
    # plt.xlabel("u")
    # plt.ylabel("y")
    # plt.show()

def podpunkt_d():
    x_valid, y_valid, x_train, y_train = podpunkt_a()
    x_matrix, y_matrix = list(), list()
    x_matrix1, x_matrix2, y_matrix1, y_matrix2 = list(), list(), list(), list()

    kp = 3
    for i in range(4):
        x_matrix.append(np.power(x_train[2:-1], i+1))
        y_matrix.append(np.power(y_train[2:-1], i+1))
        x_matrix1.append(np.power(x_train[1:-2], i+1))
        y_matrix1.append(np.power(y_train[1:-2], i+1))
        x_matrix2.append(np.power(x_train[:-3], i+1))
        y_matrix2.append(np.power(y_train[:-3], i+1))
    M = np.array(x_matrix+x_matrix1+x_matrix2+y_matrix+y_matrix1+y_matrix2).T
    y = y_train[3:]

    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(M.transpose(), M)), M.transpose()), y)

    u_value = np.linspace(-1, 1, 100)
    y_value = y_valid[:kp]
    x_matrix, y_matrix = list(), list()
    x_matrix1, x_matrix2, y_matrix1, y_matrix2 = list(), list(), list(), list()

    u_value = np.linspace(-1, 1, 100)
    y_value = []
    for u in u_value:
        y = 0
        for i in range(100):
            y = (((w[0]+w[4]+w[8])*u + (w[1]+w[5]+w[9])*u**2 + (w[2]+w[6]+w[10])*u**3 + (w[3]+w[7]+w[11])*u**4) +
            (w[12]+w[16]+w[20])*y + (w[13]+w[17]+w[21])*y**2 +(w[14]+w[18]+w[22])*y**3 + (w[15]+w[19]+w[23])*y**4)
        y_value.append(y)
    
    plt.plot(y_value)
    plt.title("Charakterystyka statyczna modelu nieliniowego")
    plt.xlabel("u")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    podpunkt_d()