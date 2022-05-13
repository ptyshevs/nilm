import numpy as np


def split_sequence2Xy(sequences):
    """Split each sequence into input tensor X and output vector y
    
    This is used for one-step-forward forecasting
    sequences is tensor of shape (n_samples, n_sequences, n_features)
    """
    n_samples, n_sequences, n_features = sequences.shape
    X = sequences[:,:n_sequences-1,:]
    y = sequences[:,n_sequences-1,:]
    return X, y

def sliding_window(X, wsize: int):
    """
    Test example:

    s = Xtr_scaled[:5, :]  
    print(sliding_window(Xtr_scaled, 5).shape)
    """
    n = len(X)
    out = []
    for i in range(n-wsize):
        out.append(X[i:i+wsize,:])
    return np.stack(out)
