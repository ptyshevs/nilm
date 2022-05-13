import plotly.express as px
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embedding(df, color=None, title=None):
    fig = px.scatter(df, x=0, y=1, color=color, title=title,
                      opacity=.6, height=600, width=600)
    fig.show()


def plot_rnn_forecast(df, title=None):
    fig = px.line(df, height=600, width=700, title=title)
    fig.update_layout(legend_orientation='h', margin={'t': 30, 'b': 0, 'l': 0, 'r': 0})
    fig.show()


def plot_sample_sequences(X, n: int = 3, columns=None, random_seed=None):
    """Plot some of the sequences"""
    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.randint(0, len(X), size=n)
    for i in idx:
        sample_seq = X[i]
        sample_seq = pd.DataFrame(sample_seq, columns=columns)
        px.line(sample_seq, width=600).show()


def plot_pca_vs_tsne(real_sequences, sample_sequences, sample_size=350, random_state=42):
    np.random.seed(random_state)
    idx = np.random.permutation(min(len(sample_sequences), len(real_sequences)))[:sample_size]

    real_sample = real_sequences[idx]
    synthetic_sample = sample_sequences[idx]

    n_samples, sequence_len, n_features = real_sample.shape
    #for the purpose of comparision we need the data to be 2-Dimensional. For that reason we are going to use only two componentes for both the PCA and TSNE.
    real_data_reduced = real_sample.reshape(-1, sequence_len)
    synth_data_reduced = synthetic_sample.reshape(-1,sequence_len)

    n_components = 2
    pca = PCA(n_components=n_components, random_state=random_state)
    tsne = TSNE(n_components=n_components, n_iter=500, learning_rate='auto', init='pca', random_state=random_state)

    #The fit of the methods must be done only using the real sequential data
    pca.fit(real_data_reduced)

    pca_real = pd.DataFrame(pca.transform(real_data_reduced))
    pca_real['label'] = 'real'
    pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))
    pca_synth['label'] = 'synth'

    pca_results = pd.concat([pca_real, pca_synth])

    data_reduced = np.concatenate((real_data_reduced, synth_data_reduced), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))
    
    plot_embedding(pca_results, color='label', title='PCA embedding')
    plot_embedding(tsne_results, color=pca_results.label, title='t-SNE embedding')
