
Questions:
1. Number of clusters according to each of the methods
2. What is the average distance of data points from the centroid of the second cluster using K-Means?
3. Calculate the silhouette score for the clusters formed using DBSCAN with  \epsilon=0.5  and  min\_samples=5 . What is the silhouette score?
4. After performing clustering using GMM, how many data points belong to the largest cluster?
5. Using DBSCAN, how many outliers (noise points) are detected when  \epsilon=0.3  and  min\_samples=10 ?
6. After applying K-Means clustering with k=3, what is the centroid value for the ‘BALANCE’ feature in the first cluster?
7. Compare the clustering results performing K Means using different distance measures. What do you think can be the best distance measure used for K Means based on the data distribution? 
8. Calculation of indicators to check for clustering ( Refer to that notebook )
9. Inferences for each of these methods - What do each of the clusters mean in each of the methods?
10. Calculate WCSS (Within-Cluster Sum of Squares) for all of these methods without using any inbuilt modules in python. 
11. Apply the cluster stability by follows:
	1. Apply KMeans method to the same dataset with different random initializations
	2. Calculate the inertia value, and obtain the standard deviation of these values
	What is the interpretation? Are the clusters obtained by using K Means algorithm stable?


Objectives:
1. Data Exploration:
	1. Data Cleaning
	2. Exploratory Data Analysis
	3. Data Visualisation
2. Clustering model: Find optimal cluster number using silhoutte score, and scores relevant to these models 
	1. K Means 
	2. GMM 
	3. DBSCAN
	4. Hierarchical Clustering
3. Post Model Analysis: Interpret the cluster meaning each of the clustering models find.
4. Compute Within Cluster, Between Cluster and Overall Variance
