library(tidyverse)
library(igraph)

## Fetch the comtrade dataset
comtrade <- read.csv('https://llperezp-datasets.s3.us-east-2.amazonaws.com/comtrade_2019.csv')

## Compute flow matrix
flow_mtx <- comtrade %>%
    ## reduce set of partners to be equal to set of reporters
    filter(Partner %in% Reporter) %>%
    ## remove self-flows (e.g. re-imports)
    filter(Partner != Reporter) %>%
    ## compute total flows for each (reporter,partner) pair
    group_by(Reporter, Partner) %>% 
    summarize(trade=sum(Trade.Value..US..)) %>%
    ## pivot into matrix, replace NAs with zeros and order alphabetically
    pivot_wider(names_from = Partner, values_from=trade) %>% 
    replace(is.na(.), 0) %>%
    arrange(-desc(Reporter)) %>% 
    column_to_rownames(var='Reporter') %>% 
    select(order(colnames(.))) %>%
    as.matrix(.)

## Make flow matrix symmetric by setting A[i,j] = max{A[i,j], A[j,i]}
ux <- pmax(flow_mtx[lower.tri(flow_mtx, diag=TRUE)], flow_mtx[lower.tri(flow_mtx, diag=TRUE)]) 
flow_mtx[lower.tri(flow_mtx, diag=TRUE)] <- ux
flow_mtx <- t(flow_mtx)
flow_mtx[lower.tri(flow_mtx, diag=TRUE)] <- ux

## Turn flow matrix into distance matrix
## log-transform to avoid heavy tail effects
dx <- 1 - log(flow_mtx + 1)/max(log(flow_mtx + 1))
write.table(dx, 'comtrade_distmat.csv', sep=',')

png('hist_distmat.png')
hist(dx[dx < 1])
dev.off()

## Hierarchical clustering
png('hclust_flowdist.png', width=2560, height=1920, pointsize=46)
par(mar=rep(0,4), omi=rep(0,4), mgp=rep(0,3))
plot(hclust(as.dist(dx), method='ward.D2'), cex=0.5, main=NULL, axes=FALSE, ylab=NULL, xlab="", sub="", hang=0.1)
dev.off()

## Minimum Spanning Tree
tree <- mst(graph_from_adjacency_matrix(dx, weighted = TRUE, mode='undirected'), algorithm='prim')
png('mst_flowdist.png', width=2560, height=1920, pointsize=46)
par(mar=rep(0,4), omi=rep(0,4), mgp=rep(0,3))
plot(tree, vertex.size=0.5, vertex.label.cex=0.4, vertex.label.family='sans', vertex.label.color='black', edge.curved=TRUE, edge.color='red', edge.width=2, layout=layout.lgl, margin=rep(0, 4))
dev.off()
