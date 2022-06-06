import numpy as np

def load_data_numpy(file_x, file_y, file_label, file_tx, file_ty, file_tlabel, my_seed = 123):
    
    X = np.load(file_x)
    X = X.astype(np.long)
    Y = np.load(file_y)
    L = np.load(file_label)

    tX = np.load(file_tx)
    tX = tX.astype(np.long)
    tY = np.load(file_ty)
    tL = np.load(file_tlabel)

    X = np.concatenate((X, tX), axis=0)
    Y = np.concatenate((Y, tY), axis=0)
    L = np.concatenate((L, tL), axis=0)

    np.random.seed(my_seed)

    rnd_indx = np.random.choice(range(Y.shape[0]), size=Y.shape[0])

    Yt = Y[rnd_indx] #.reshape((1, Y.shape[0]))
    Xt = X[rnd_indx]
    Lt = L[rnd_indx]

    return Xt, Yt, Lt
