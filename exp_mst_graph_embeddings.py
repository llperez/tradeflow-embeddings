import comtrade as ct
import pmfg
from networkx.algorithms import minimum_spanning_tree

## get data
df = ct.comtrade_df()
A = ct.adj_matrix(df)
Ax = A.to_numpy()
Dx = 1.0 - ct.rescale(ct.make_symmetric(Ax))

G = ct.build_graph(A.index, Dx)

## compute MST
MST = minimum_spanning_tree(G)

## build list of edges from MST for StellarGraph
SG = ct.build_stellargraph(list(MST.edges()), A.index, 1.0 - Dx)
node_ids, embeddings = ct.compute_embeddings(SG, out_file='exp_mst_graph_country_embeddings.csv')
ct.compute_tsne(embeddings, node_ids, out_file='exp_mst_graph_tsne_embeddings.png')
