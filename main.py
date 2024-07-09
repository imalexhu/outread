
from Cluster import Cluster
from DataProcessing import DataProcessing
from Vectorise import Vectorise
from Output import Output

config = {
    # "DataProcessing" : "Stemming" ,
    "DataProcessing" : "Lemmatization",
    # "Vectorise" : "TF-IDF",
    "Vectorise": "Word2Vec",
    "Cluster" : "KMeans",
    "Visualise" : "PCA",
    # "Visualise" : "TSNE",
}



def main():
    # preprocess the Gree Energy Datasets
    processor = DataProcessing(config, 'Green Energy Dataset')
    vectoriser = Vectorise(config,processor)
    clusterisor = Cluster(config,vectoriser).cluster()
    Output(config,clusterisor, vectoriser).generate()


if __name__ == "__main__":
    main()



