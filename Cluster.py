import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from Vectorise import Vectorise
import pandas as pd
import numpy as np

class Cluster:
    def __init__(self, config, vectoriser : Vectorise):
        self.vectoriser = vectoriser
        self.vectorised_data = vectoriser.vectorised_results
        self.config = config
    
    def get_optimal_clusters(self):
        if self.config['Vectorise'] == 'Word2Vec':
            return 3
        else :
            return 9
    
    def get_cluster_top_terms(self,cluster_labels):
        if self.config['Vectorise'] == 'Word2Vec':
            """Get the top terms for each cluster based on term frequency."""
            clusters = {}
            
            for doc, label in zip(self.vectoriser.preprocessed_data.items(), cluster_labels):
                name = doc[0]
                split_words = doc[1].split()
                if label not in clusters:
                    clusters[label] = []
                for word in split_words:
                    clusters[label].append(word)
            
            top_terms = {}
            for cluster, words in clusters.items():
                
                word_counts = Counter(words)
                print("For cluster " + str(cluster + 1) + " the top terms are:")
                top_terms[cluster] = [word for word, count in word_counts.most_common(20)]
            return top_terms
        else :
            terms = self.vectoriser.tf_idf.get_feature_names_out()
            top_terms = self.get_top_terms_per_cluster(self.vectoriser.tf_idf_matrix, cluster_labels, terms)
            return top_terms
    
    def get_top_terms_per_cluster(self,tfidf, cluster_labels, terms, top_n=20):
        """Get the top terms for each cluster based on TF-IDF scores."""
        df = pd.DataFrame(tfidf.todense()).groupby(cluster_labels).mean()
        top_terms = {}
        for i, row in df.iterrows():
            top_terms[i] = [terms[t] for t in np.argsort(row)[-top_n:]]
        return top_terms
    
    def cluster(self):
        kmeans = KMeans(n_clusters=self.get_optimal_clusters(), random_state=42)
        cluster_labels = kmeans.fit_predict(list(self.vectorised_data.values()))
        # Compute evaluation metrics
        silhouette_avg = silhouette_score(list(self.vectorised_data.values()), cluster_labels)
        davies_bouldin = davies_bouldin_score(list(self.vectorised_data.values()), cluster_labels)

        print(f"Silhouette Score: {silhouette_avg}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")
        
        # Save clustering results to a file
        documents_flat = ["".join(doc) for doc in self.vectorised_data.keys()]
        
        # print(documents_flat)
        results_df = pd.DataFrame({'Document': documents_flat, 'Cluster': cluster_labels})
        results_df.to_csv('clustering_results.csv', index=False)
                
        # Generate a summary report
        summary = {
            'Number of Clusters': self.get_optimal_clusters(),
            'Silhouette Score': silhouette_avg,
            'Davies-Bouldin Index': davies_bouldin
        }

        cluster_summary = results_df.groupby('Cluster').size().reset_index(name='Number of Papers')
        summary_df = pd.DataFrame(summary, index=[0])
        summary_df = summary_df.join(cluster_summary.set_index('Cluster').T, how='outer').T
        
        
        summary_df = pd.DataFrame(summary, index=[0])
        summary_df = summary_df.join(cluster_summary.set_index('Cluster').T, how='outer').T
        
        print("Summary Report:")
        print(summary_df)
        
        
        top_terms = self.get_cluster_top_terms(cluster_labels)

        
        # Save the summary report and key terms to a file
        with open('summary_report.txt', 'w') as f:
            f.write(f"Number of Clusters: {self.get_optimal_clusters()}\n")
            f.write(f"Silhouette Score: {silhouette_avg}\n")
            f.write(f"Davies-Bouldin Index: {davies_bouldin}\n\n")
            f.write("Number of Papers in Each Cluster:\n")
            for cluster, size in cluster_summary.itertuples(index=False):
                f.write(f"Cluster {cluster + 1}: {size} papers\n")
            f.write("\nKey Terms for Each Cluster:\n")
            for cluster, terms in top_terms.items():
                f.write(f"Cluster {cluster + 1}: {', '.join(terms)}\n")

        input()

        return kmeans
        
        # stolen from chat-gpt
    def cluster_data_elbow(self):
        wcss = []
        max_clusters = 10
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(list(self.vectorised_data.values()))
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method For Optimal Number of Clusters')
        plt.savefig('elbow_w.png')
        plt.show()
    pass
            
    # stolen from chat-gpt
    def cluster_data_silhouette(self) : 
        silhouette_scores = []
        max_clusters = 10

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(list(self.vectorised_data.values()))
            silhouette_avg = silhouette_score(list(self.vectorised_data.values()), labels=cluster_labels)
            silhouette_scores.append(silhouette_avg)


        silhouette_avg = silhouette_score(list(self.vectorised_data.values()), cluster_labels)
        davies_bouldin = davies_bouldin_score(list(self.vectorised_data.values()), cluster_labels)

        print(f"Silhouette Score: {silhouette_avg}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")

        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis For Optimal Number of Clusters')
        # plt.savefig('silhouette_w.png')
        

        # Optimal number of clusters
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        print(f"Optimal number of clusters based on silhouette analysis: {optimal_clusters}")