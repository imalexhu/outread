
from DataProcessing import DataProcessing
# from Vectorise import Vectorise
# from Output import Output
# from Visualise import Visualise

config = {
    # "DataProcessing" : "Stemming" ,
    "DataProcessing" : "Lemmatization",
    "Vectorise" : "TF-IDF",
    # "Vectorise": "Word2Vec",
}



def main():
    # preprocess the Gree Energy Datasets
    processed_results = DataProcessing(config, 'Green Energy Dataset').process_texts()
    # vectoried_results = Vectorise(config,processed_results).vectorise_data()
    # clustered_results = Cluster(vectoried_results,config).cluster_data()
    # Output(clustered_results).genereate()
    # Visualise(clustered_result).generate()


if __name__ == "__main__":
    main()



