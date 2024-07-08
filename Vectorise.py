import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


class Vectorise:
    def __init__(self, config, data : dict[str,str]):
        self.config = config
        self.preprocessed_data = data
    
    
    def vectorise_data(self) -> np.ndarray:
        result : dict[str:np.ndarray] = {} 
        if(self.config["Vectorise"] == "TF-IDF") :
            # TF-IDF Vectorization
            # generate vectors for each document
            vectoriser = TfidfVectorizer()
            tf_idf = vectoriser.fit_transform(self.preprocessed_data.values())
            tf_idf.mean(axis=1)
            idx = 0
            for key, value in self.preprocessed_data.items():
                result[key] = np.array(tf_idf.mean(axis=1)[idx])
                idx += 1
        else :
            # Word2Vec Vectorization
            text = list(self.preprocessed_data.values())
            model = Word2Vec(text, vector_size=100, window=5, min_count=1, workers=4)
            # Generate vectors for each document
            for(key, value) in self.preprocessed_data.items():
                words = value.split()
                vec = np.mean([model.wv[word] for word in words if word in model.wv], axis=0)
                result[key] = np.array(vec)
        return result            
