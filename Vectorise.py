import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from DataProcessing import DataProcessing


class Vectorise:
    def __init__(self, config, processor : DataProcessing):
        self.config = config
        self.preprocessed_data = processor.processed_results
        self.word2vec_model = None
        self.tf_idf = None
        self.tf_idf_matrix = None
        self.vectorised_results = self.vectorise_data()
        
    
    def vectorise_data(self) -> dict[str, np.ndarray]:
        result : dict[str:np.ndarray] = {} 
        if(self.config["Vectorise"] == "TF-IDF") :
            # TF-IDF Vectorization
            # generate vectors for each document
            self.tf_idf = TfidfVectorizer()
            self.tf_idf_matrix = self.tf_idf.fit_transform(self.preprocessed_data.values())
            result = {key: value for key, value in zip(self.preprocessed_data.keys(), self.tf_idf_matrix.toarray())}
        else :
            # Word2Vec Vectorization
            text = list(self.preprocessed_data.values())
            self.word2vec_model = Word2Vec(text, vector_size=100, window=5, min_count=1, workers=4)
            # Generate vectors for each document
            for(key, value) in self.preprocessed_data.items():
                words = value.split()
                vec = np.mean([self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv], axis=0)
                result[key] = np.array(vec)
                
                
            
        return result            
