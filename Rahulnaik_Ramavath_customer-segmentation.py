import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme() 
sns.set_palette("husl")

print("Loading datasets...")
customers_df = pd.read_csv('Customers.csv')
transactions_df = pd.read_csv('Transactions.csv')


customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

class CustomerSegmentation:
    def __init__(self, customers_df, transactions_df):
        self.customers_df = customers_df
        self.transactions_df = transactions_df
        self.features = None
        self.features_scaled = None
        self.clusters = None
        
    def create_features(self):
        print("Creating customer features...")
        # Calculate RFM metrics
        current_date = self.transactions_df['TransactionDate'].max()
        
        rfm = self.transactions_df.groupby('CustomerID').agg({
            'TransactionDate': lambda x: (current_date - x.max()).days,  
            'TransactionID': 'count',  
            'TotalValue': 'sum' 
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        print("Calculating additional features...")
        avg_basket = self.transactions_df.groupby('CustomerID')['TotalValue'].mean()
        purchase_variance = self.transactions_df.groupby('CustomerID')['TotalValue'].std()
        
        purchase_dates = self.transactions_df.groupby('CustomerID')['TransactionDate'].agg(['min', 'max'])
        purchase_dates['customer_lifetime'] = (purchase_dates['max'] - purchase_dates['min']).dt.days
        
        purchase_dates['avg_purchase_frequency'] = purchase_dates['customer_lifetime'] / rfm['Frequency']
        
        self.features = pd.concat([
            rfm,
            avg_basket.rename('AvgBasket'),
            purchase_variance.rename('PurchaseVariance'),
            purchase_dates[['customer_lifetime', 'avg_purchase_frequency']]
        ], axis=1).fillna(0)
        
        print("Scaling features...")
        scaler = StandardScaler()
        self.features_scaled = pd.DataFrame(
            scaler.fit_transform(self.features),
            index=self.features.index,
            columns=self.features.columns
        )
        
    def find_optimal_clusters(self, max_clusters=10):
        print("Finding optimal number of clusters...")
        if self.features_scaled is None:
            self.create_features()
            
        db_scores = []
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            print(f"Testing {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.features_scaled)
            
            db_score = davies_bouldin_score(self.features_scaled, clusters)
            db_scores.append(db_score)
            
            silhouette_avg = silhouette_score(self.features_scaled, clusters)
            silhouette_scores.append(silhouette_avg)
            
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), db_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Davies-Bouldin Score')
        plt.title('Davies-Bouldin Score vs Number of Clusters')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        
        plt.tight_layout()
        plt.show()
        
        optimal_clusters = np.argmin(db_scores) + 2
        return optimal_clusters, db_scores, silhouette_scores
    
    def perform_clustering(self, n_clusters):
        print(f"Performing clustering with {n_clusters} clusters...")
        if self.features_scaled is None:
            self.create_features()
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(self.features_scaled)
        
        self.features['Cluster'] = self.clusters
        
        db_index = davies_bouldin_score(self.features_scaled, self.clusters)
        silhouette_avg = silhouette_score(self.features_scaled, self.clusters)
        
        print(f"\nClustering Metrics:")
        print(f"Davies-Bouldin Index: {db_index:.4f}")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        return db_index, silhouette_avg
    
    def analyze_clusters(self):
        print("\nAnalyzing clusters...")
        cluster_stats = self.features.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'AvgBasket': 'mean',
            'customer_lifetime': 'mean'
        }).round(2)
        
        print("\nCluster Statistics:")
        print(cluster_stats)
        
        return cluster_stats
    
    def visualize_clusters(self):
        print("\nGenerating cluster visualizations...")
        plt.figure(figsize=(20, 15))
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(
            data=self.features,
            x='Recency',
            y='Monetary',
            hue='Cluster',
            palette='deep'
        )
        plt.title('Recency vs Monetary by Cluster')
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(
            data=self.features,
            x='Frequency',
            y='Monetary',
            hue='Cluster',
            palette='deep'
        )
        plt.title('Frequency vs Monetary by Cluster')
        
        plt.subplot(2, 2, 3)
        cluster_sizes = self.features['Cluster'].value_counts().sort_index()
        cluster_sizes.plot(kind='bar')
        plt.title('Cluster Sizes')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        
        plt.subplot(2, 2, 4)
        cluster_means = self.features.groupby('Cluster').mean()
        sns.heatmap(cluster_means, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Average Features by Cluster')
        
        plt.tight_layout()
        plt.show()

print("Starting customer segmentation analysis...")
segmentation = CustomerSegmentation(customers_df, transactions_df)

optimal_clusters, db_scores, silhouette_scores = segmentation.find_optimal_clusters()
print(f"\nOptimal number of clusters: {optimal_clusters}")

db_index, silhouette_avg = segmentation.perform_clustering(optimal_clusters)

cluster_stats = segmentation.analyze_clusters()

segmentation.visualize_clusters()

results_df = pd.DataFrame({
    'CustomerID': segmentation.features.index,
    'Cluster': segmentation.features['Cluster']
})

output_filename = 'FirstName_LastName_Clustering.csv'
results_df.to_csv(output_filename, index=False)
print(f"\nResults saved to {output_filename}")

print("\nClustering Summary for PDF Report:")
print(f"Number of clusters: {optimal_clusters}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print("\nCluster Sizes:")
print(results_df['Cluster'].value_counts().sort_index())