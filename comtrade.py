## Routines for loading and preparing the Comtrade raw dataset,
## transforming it into an adjacency/dissimilarity matrix and
## computing graph embeddings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from stellargraph import StellarGraph
from stellargraph import StellarDiGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def comtrade_df():
    df = pd.read_csv('https://llperezp-datasets.s3.us-east-2.amazonaws.com/comtrade_2019.csv')

    # project columns
    df = df[['Reporter', 'Partner', 'Trade Flow', 'Trade Value (US$)']]
    df.columns = ['reporter', 'partner', 'trade_flow', 'trade_value']

    # filter out partners not reported, self-flows, etc. 
    reporters = set(df['reporter'])
    df = df.loc[df['partner'].isin(reporters)]
    df = df.loc[df['reporter'] != df['partner']]
    return df


def adj_matrix(df, trade_flows=None):
    if trade_flows is not None:
        df = df.loc[df['trade_flow'].isin(trade_flows)]

    # group-by, aggregate and pivot table into matrix 
    dfg = df.groupby(['reporter', 'partner']).sum('trade_value')
    dfp = dfg.pivot_table(index='reporter', columns='partner')
    return dfp.fillna(0)


def make_symmetric(Ax):
    L = np.maximum(np.triu(Ax).T, np.tril(Ax))
    il = np.tril_indices(Ax.shape[0])
    U = L.T
    U[il] = L[il]
    return U


def rescale(Ax, axis=None, func=lambda x: np.log(x + 1)):    
    A_scaled = func(Ax) if func is not None else Ax
    A_max = A_scaled.max(axis=axis)
    return A_scaled / A_max


def build_graph(names, Dx, directed=False):
    G = nx.Graph() if not directed else nx.DiGraph()
    for (i, row) in enumerate(Dx):
        for (j, weight) in enumerate(row):
            G.add_edge(names[i], names[j], weight=weight)

    return G


def build_stellargraph(edge_list, names, Dx, directed=False):
    Dx = pd.DataFrame(Dx, index=names, columns=names)
    Gx = pd.DataFrame([{ 'source': n1, 'target': n2, 'weight': Dx[n1][n2] } for (n1,n2) in edge_list ])
    G = StellarGraph(edges=Gx) if not directed else StellarDiGraph(edges=Gx)
    return G


def compute_embeddings(G, out_file='country_embeddings.csv', walk_length=100, num_walks=50, rw_p=0.5, rw_q=2.0, w2v_size=20, w2v_window=6, rng_seed=31337):
    rw = BiasedRandomWalk(G)
    weighted_walks = rw.run(
        nodes=G.nodes(),
        length=walk_length,
        n=num_walks,
        p=rw_p,
        q=rw_q,
        weighted=True,
        seed=rng_seed
    )

    weighted_model = Word2Vec(
        weighted_walks,
        size=w2v_size,
        window=w2v_window,
        min_count=0,
        sg=1,
        workers=1,
        iter=1
    )

    node_ids = weighted_model.wv.index2word
    embeddings = (weighted_model.wv.vectors)

    embed_frame = pd.DataFrame(embeddings, index=node_ids)
    embed_frame.to_csv(out_file, header=False)
    
    return (node_ids, embeddings)


def compute_tsne(node_embeddings, node_ids, out_file='tsne_embeddings.png', rng_seed=31337):
    tsne = TSNE(n_components=2, random_state=rng_seed)
    embed = tsne.fit_transform(node_embeddings)
    fig, ax = plt.subplots(figsize=(16, 15))
    ax.scatter(embed[:, 0], embed[:, 1], cmap="jet", alpha=0.7)
    for i, country in enumerate(node_ids):
        ax.annotate(country, (embed[i, 0], embed[i, 1]))

    plt.savefig(out_file, dpi=300)
    
    return embed


