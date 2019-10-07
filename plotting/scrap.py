import numpy as np

if __name__ == "__main__":
    n_sen = np.array([10, 10, 10, 10, 10, 10, 10, 10, 8, 11])
    n_sen_mean = n_sen.mean()
    n_sen_std = n_sen.std()

    print(n_sen_mean, n_sen_std)
