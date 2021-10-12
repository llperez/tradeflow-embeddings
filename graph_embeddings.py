## 'default' embedding configuration: undirected graph, filtered by PMFG
import comtrade as ct
import pmfg

## get data
df = ct.comtrade_df()
A = ct.adj_matrix(df)
Ax = A.to_numpy()
Dx = 1.0 - ct.rescale(ct.make_symmetric(Ax))

G = ct.build_graph(A.index, Dx)

## compute PMFG
sorted_edges = pmfg.sort_graph_edges(G)
PMFG = pmfg.compute_PMFG(sorted_edges, len(G))

## build list of edges from PMFG for StellarGraph
## note: 1-weight for biased random walk prob
SG = ct.build_stellargraph(list(PMFG.edges), A.index, 1.0 - Dx)
node_ids, embeddings = ct.compute_embeddings(SG)
ct.compute_tsne(embeddings, node_ids)
