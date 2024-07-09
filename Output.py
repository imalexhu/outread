from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from Cluster import Cluster
from Vectorise import Vectorise
class Output:
    def __init__(self, config, clusterisor : Cluster, vectoriser : Vectorise):
        self.config = config
        self.cluster = clusterisor
        self.vectorised_data = vectoriser.vectorised_results
        pass
    
    def get_optimal_clusters(self):
        if self.config['Vectorise'] == 'Word2Vec':
            return 3
        else :
            return 9
    
    
    def generate(self):
        if self.config['Visualise'] == 'PCA':
            self.visualise_pca()
        else:
            self.visualise_tsne()
        pass
            
    def visualise_pca(self):
        
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(list(self.vectorised_data.values()))

        # Plot the clusters
        plt.figure(figsize=(10, 6))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Define colors for different clusters

        for i in range(self.get_optimal_clusters()):
            points = reduced_data[i]
            plt.scatter(points[0], points[ 1], s=50, c=colors[i % len(colors)], label=f'Cluster {i+1}')

        plt.title('K-means Clustering Results')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()
        plt.savefig("pca_w.png")
        
        pass
    
    def visualise_tsne(self):
        pass