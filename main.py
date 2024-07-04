
from DataProcessing import DataProcessing

config = {
    "DataProcessing" : "Stemming" 
    # "DataProcessing" : "Lemmatization
}



def main():
    # preprocess the Gree Energy Datasets
    dp = DataProcessing(config, 'Green Energy Dataset').process_texts()


if __name__ == "__main__":
    main()



