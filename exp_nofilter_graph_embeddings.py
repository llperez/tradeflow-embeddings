import comtrade as ct
import pmfg

## get data
df = ct.comtrade_df()
A = ct.adj_matrix(df)
Ax = A.to_numpy()
Dx = ct.rescale(ct.make_symmetric(Ax))

G = ct.build_graph(A.index, Dx)

## build list of edges from G for StellarGraph
SG = ct.build_stellargraph(list(G.edges()), A.index, Dx)
node_ids, embeddings = ct.compute_embeddings(SG, out_file='exp_nofilter_graph_country_embeddings.csv')
ct.compute_tsne(embeddings, node_ids, out_file='exp_nofilter_graph_tsne_embeddings.png')
