import pandas as pd
import networkx as nx
import pmfg
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

## constants
walk_length = 150
num_walks = 50
rw_p = 0.5
rw_q = 2.0
rng_seed = 31337
w2v_size = 50
w2v_window = 5

## build graph from matrix
dmtx = pd.read_csv('comtrade_distmat.csv')
gx = nx.Graph()
for (i, rows) in dmtx.iterrows():
    for (j, wt) in rows.iteritems():
        gx.add_edge(i, j, weight=wt)

## compute PMFG
sorted_edges = pmfg.sort_graph_edges(gx)
PMFG = pmfg.compute_PMFG(sorted_edges, len(gx.nodes))

## build list of edges from PMFG for StellarGraph
## note: 1-weight for biased random walk prob
list_edges = [ { 'source': n1, 'target': n2, 'weight': 1-dmtx[n1][n2] } for (n1,n2) in list(PMFG.edges) ] 
G = StellarGraph(edges=pd.DataFrame(list_edges))
rw = BiasedRandomWalk(G)

weighted_walks = rw.run(
    nodes=G.nodes(),
    length=walk_length,
    n=num_walks,
    p=rw_p,
    q=rw_q,
    weighted=True,
    seed=rng_seed)

weighted_model = Word2Vec(
    weighted_walks,
    size=w2v_size,
    window=w2v_window,
    min_count=0,
    sg=1,
    workers=1,
    iter=1)

node_ids = weighted_model.wv.index2word
weighted_node_embeddings = (weighted_model.wv.vectors)

tsne = TSNE(n_components=2, random_state=rng_seed)
weighted_node_embeddings_2d = tsne.fit_transform(weighted_node_embeddings)

fig, ax = plt.subplots(figsize=(16, 15))
ax.scatter(
    weighted_node_embeddings_2d[:, 0],
    weighted_node_embeddings_2d[:, 1],
    cmap="jet",
    alpha=0.7)

for i, country in enumerate(node_ids):
    ax.annotate(country,
                (weighted_node_embeddings_2d[i, 0],
                 weighted_node_embeddings_2d[i, 1]))
    
plt.savefig('tsne_embeddings.png', dpi=150)

embed_frame = pd.DataFrame(weighted_node_embeddings, index=node_ids)
embed_frame.to_csv('country_embeddings.csv', header=False)
