# Hierarchical-Clustering
.Make a segment of mall dataset using Hierarchical Clustering (Mall Customers Hierarchical Clustering)


# What is Clustering??

Clustering is a technique that groups similar objects such that the objects in the same group are more similar to each other than the objects in the other groups. The group of similar objects is called a Cluster.

![clustered data point](https://user-images.githubusercontent.com/29980448/82360598-3f064900-9a27-11ea-836f-28697430d89f.jpg)

There are 5 popular clustering algorithms that data scientists need to know:

1. K-Means Clustering.
2. Hierarchical Clustering: We’ll discuss this algorithm here in detail
3. Mean-Shift Clustering.
4. Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
5. Expectation-Maximization (EM) Clustering using Gaussian Mixture Models (GMM)

# Hierarchical Clustering Algorithm

Also called Hierarchical cluster analysis or HCA is an unsupervised clustering algorithm which involves creating clusters that have predominant ordering from top to bottom.

For e.g: All files and folders on our hard disk are organized in a hierarchy.

The algorithm groups similar objects into groups called clusters. The endpoint is a set of clusters or groups, where each cluster is distinct from each other cluster, and the objects within each cluster are broadly similar to each other.

This clustering technique is divided into two types:

1. Agglomerative Hierarchical Clustering.

2. Divisive Hierarchical Clustering.


# Agglomerative Hierarchical Clustering
 
The Agglomerative Hierarchical Clustering is the most common type of hierarchical clustering used to group objects in clusters based on their similarity. It’s also known as AGNES (Agglomerative Nesting). It's a “bottom-up” approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

# How does it work?

1. Make each data point a single-point cluster → forms N clusters
2. Take the two closest data points and make them one cluster → forms N-1 clusters
3. Take the two closest clusters and make them one cluster → Forms N-2 clusters.
4. Repeat step-3 until you are left with only one cluster.

Have a look at the visual representation of Agglomerative Hierarchical Clustering for better understanding:
![Agglomerative hierarchical clustering](https://user-images.githubusercontent.com/29980448/82361286-44b05e80-9a28-11ea-988e-82ec4a4aad1b.gif)


There are several ways to measure the distance between clusters in order to decide the rules for clustering, and they are often called Linkage Methods. Some of the common linkage methods are:

. Complete-linkage: the distance between two clusters is defined as the longest distance between two points in each cluster.

. Single-linkage: the distance between two clusters is defined as the shortest distance between two points in each cluster. This linkage may be used to detect high values in your dataset which may be outliers as they will be merged at the end.

. Average-linkage: the distance between two clusters is defined as the average distance between each point in one cluster to every point in the other cluster.

. Centroid-linkage: finds the centroid of cluster 1 and centroid of cluster 2, and then calculates the distance between the two before merging.

The choice of linkage method entirely depends on you and there is no hard and fast method that will always give you good results. Different linkage methods lead to different clusters.

The point of doing all this is to demonstrate the way hierarchical clustering works, it maintains a memory of how we went through this process and that memory is stored in Dendrogram.


# What is a Dendrogram?

A Dendrogram is a type of tree diagram showing hierarchical relationships between different sets of data.

As already said a Dendrogram contains the memory of hierarchical clustering algorithm, so just by looking at the Dendrgram you can tell how the cluster is formed.

![Dendrogram](https://user-images.githubusercontent.com/29980448/82361804-ffd8f780-9a28-11ea-9290-5db6e9d2076e.gif)

# Note:- 

1. Distance between data points represents dissimilarities.

2. Height of the blocks represents the distance between clusters.

So you can observe from the above figure that initially P5 and P6 which are closest to each other by any other point are combined into one cluster followed by P4 getting merged into the same cluster(C2). Then P1and P2 gets combined into one cluster followed by P0 getting merged into the same cluster(C4). Now P3 gets merged in cluster C2 and finally, both clusters get merged into one.

# Parts of a Dendrogram

![0_ESGWAWTMwZi_xTz-](https://user-images.githubusercontent.com/29980448/82362220-a58c6680-9a29-11ea-9a4f-bdfb8f4d19a7.png)

A dendrogram can be a column graph (as in the image below) or a row graph. Some dendrograms are circular or have a fluid-shape, but the software will usually produce a row or column graph. No matter what the shape, the basic graph comprises the same parts:

  . The Clades are the branch and are arranged according to how similar (or dissimilar) they are. Clades that are close to the same height are similar to each other; clades with different heights are dissimilar — the greater the difference in height, the more dissimilarity.

  . Each clade has one or more #leaves.
  
  . Leaves A, B, and C are more similar to each other than they are to leaves D, E, or F.
  
  . Leaves D and E are more similar to each other than they are to leaves A, B, C, or F.
 
  . Leaf F is substantially different from all of the other leaves.
  
  A clade can theoretically have an infinite amount of leaves. However, the more leaves you have, the harder the graph will be to read with the naked eye.

# One question that might have intrigued you by now is how do you decide when to stop merging the clusters?

You cut the dendrogram tree with a horizontal line at a height where the line can traverse the maximum distance up and down without intersecting the merging point.

For example in the below figure L3 can traverse maximum distance up and down without intersecting the merging points. So we draw a horizontal line and the number of verticle lines it intersects is the optimal number of clusters.

![chosing optimum no of cluster](https://user-images.githubusercontent.com/29980448/82362888-ad98d600-9a2a-11ea-8cfd-5017de188db3.jpg)
# Number of Clusters in this case = 3.

# Divisive Hierarchical Clustering

In Divisive or DIANA(DIvisive ANAlysis Clustering) is a top-down clustering method where we assign all of the observations to a single cluster and then partition the cluster to two least similar clusters. Finally, we proceed recursively on each cluster until there is one cluster for each observation. So this clustering approach is exactly opposite to Agglomerative clustering.
![Divisive Hierarchical clustering](https://user-images.githubusercontent.com/29980448/82363107-02d4e780-9a2b-11ea-9482-37687a569e58.jpg)

There is evidence that divisive algorithms produce more accurate hierarchies than agglomerative algorithms in some circumstances but is conceptually more complex.

In both agglomerative and divisive hierarchical clustering, users need to specify the desired number of clusters as a termination condition(when to stop merging).

 




  
  


