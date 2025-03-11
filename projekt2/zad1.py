import matplotlib.pyplot as plt
import numpy as np

def podpunkt_a():
    with open('data/danestat70.txt',  'r') as data:
        x_train = list()
        x_valid = list()
        y_train = list()
        y_valid = list()
        for i, line in enumerate(data):
            p = line.split()
            if i%3 == 0:
                x_valid.append(float(p[0]))
                y_valid.append(float(p[1]))
            else: 
                x_train.append(float(p[0]))
                y_train.append(float(p[1]))
    # 67% treningowe, 33% - walidacyjne
    # plt.scatter(x_train, y_train)
    # plt.title("Dane uczące")
    # plt.scatter(x_valid, y_valid)
    # plt.title("Dane weryfikujące")
    # plt.xlabel("u")
    # plt.ylabel("y")
    # plt.show()
    
    return np.array(x_valid), np.array(y_valid), np.array(x_train), np.array(y_train)

def podpunkt_b():
    x_valid, y_valid, x_train, y_train = podpunkt_a()
    
    M = np.vstack([x_train, np.ones(len(x_train))]).T

    weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(M.transpose(), M)), M.transpose()), y_train)

    y_mod_train = weights[1] + weights[0]*x_train
    y_mod_valid = weights[1] + weights[0]*x_valid

    # Liczenie błędów
    mse_train, sse_train = float(), float()
    mse_valid, sse_valid = float(), float()

    mse_train = np.mean((y_mod_train - y_train)**2)
    sse_train = np.sum((y_mod_train - y_train)**2)

    mse_valid = np.mean((y_mod_valid - y_valid)**2)
    sse_valid = np.sum((y_mod_valid - y_valid)**2)

    print(f"Dane uczące: {mse_train:.4f}, {sse_train:.4f} \n Dane weryfikacyjne: {mse_valid:.4f}, {sse_valid:.4f}")
    # Wykresy
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, c='blue', label='Zbiór uczący')
    plt.plot(np.sort(x_train), np.sort(y_mod_train), color='red', label='Charakterystyka statyczna y(u)')
    plt.title('Charakterystyka statyczna y(u) na tle zbioru uczącego')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(x_valid, y_valid, c='red', label='Zbiór weryfikujący')
    plt.plot(np.sort(x_valid), np.sort(y_mod_valid), color='green', label='Charakterystyka statyczna y(u)')
    plt.title('Charakterystyka statyczna y(u) na tle zbioru weryfikującego')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

def weights_print(weights):
    output = str()
    for i, w in enumerate(weights):
        if i == 0:
            output += f"{w:.2f} "
        else:
            output += f"+ {w:.2f}*u^{i}"
    return output

def podpunkt_c(N):
    x_valid, y_valid, x_train, y_train = podpunkt_a()

    x_degree = list()

    for n in range(N):
        x_degree.append(pow(x_train, n+1))
    x = [np.ones(len(x_train))]
    x.append(x_degree)
    M = np.vstack(x).T
    
    weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(M.transpose(), M)), M.transpose()), y_train)

    y_mod_train, y_mod_valid = 0, 0
    for i, weight in enumerate(weights):
        y_mod_train += weight * x_train**i
        y_mod_valid += weight * x_valid**i
    
    # Liczenie błędów
    mse_train, sse_train = float(), float()
    mse_valid, sse_valid = float(), float()

    mse_train = np.mean((y_mod_train - y_train)**2)
    sse_train = np.sum((y_mod_train - y_train)**2)

    mse_valid = np.mean((y_mod_valid - y_valid)**2)
    sse_valid = np.sum((y_mod_valid - y_valid)**2)

    print(f"Dane uczące: {mse_train:.4f}, {sse_train:.4f} \n Dane weryfikacyjne: {mse_valid:.4f}, {sse_valid:.4f}")
    print(weights_print(weights))
    # Wykresy
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, c='blue', label='Zbiór uczący')
    plt.plot(np.sort(x_train), np.sort(y_mod_train), color='red', label='Charakterystyka statyczna y(u)')
    plt.title('Charakterystyka statyczna y(u) na tle zbioru uczącego')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(x_valid, y_valid, c='red', label='Zbiór weryfikujący')
    plt.plot(np.sort(x_valid), np.sort(y_mod_valid), color='green', label='Charakterystyka statyczna y(u)')
    plt.title('Charakterystyka statyczna y(u) na tle zbioru weryfikującego')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
   podpunkt_c(7)
   #podpunkt_c(5)
   #podpunkt_c(7)
   #podpunkt_b()