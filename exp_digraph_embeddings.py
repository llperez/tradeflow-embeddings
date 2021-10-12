## experiment: directed graph, using exclusively the export flows,
## filtered with spanning arborescence
import comtrade as ct
import pmfg
from networkx.algorithms.tree.branchings import Edmonds

## get data -- just 'Export' flows. 
df = ct.comtrade_df()
A = ct.adj_matrix(df, ['Export'])
Ax = A.to_numpy()
Dx = 1.0 - ct.rescale(Ax)

## get the digraph
G = ct.build_graph(A.index, Dx, directed=True)

## compute the spanning arborescence
MG = Edmonds(G).find_optimum(kind='min', style='arborescence').reverse()

## build list of edges
SG = ct.build_stellargraph(list(MG.edges()), A.index, 1.0 - Dx, directed=True)
node_ids, embeddings = ct.compute_embeddings(SG, out_file='exp_digraph_country_embeddings.csv')
ct.compute_tsne(embeddings, node_ids, out_file='exp_digraph_tsne_embeddings.png')
