import comtrade as ct
import pmfg

## get data -- just 'Export' flows
df = ct.comtrade_df()
A = ct.adj_matrix(df, ['Export'])
Ax = A.to_numpy()
Dx = ct.rescale(Ax)

G = ct.build_graph(A.index, Dx, directed=True)

## build list of edges from G for StellarGraph
SG = ct.build_stellargraph(list(G.edges()), A.index, Dx, directed=True)
node_ids, embeddings = ct.compute_embeddings(SG, out_file='exp_nofilter_digraph_country_embeddings.csv')
ct.compute_tsne(embeddings, node_ids, out_file='exp_nofilter_digraph_tsne_embeddings.png')
