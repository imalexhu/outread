
from PyPDF2 import PdfReader
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class DataProcessing :
    def __init__(self, config,folder_path) -> None:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.folder_path = folder_path
        self.config = config
        self.processed_results = self.process_texts()
    
    def get_cache_path(self,pdf_file_name : str) -> str:
        file_path = ""
        
        if self.config['DataProcessing'] == 'Stemming':
            file_path = os.path.join('cache_s', pdf_file_name)
        else :
            file_path = os.path.join('cache_l', pdf_file_name)
        return file_path
    
    def has_cache(self,pdf_file_name : str) -> bool:
        return os.path.exists(self.get_cache_path(pdf_file_name))
    
    def load_from_cache(self,pdf_file_name : str) -> str:
        with open(self.get_cache_path(pdf_file_name), 'r') as file:
            text = file.read()
            return text
        
    def save_to_cache(self,pdf_file_name : str, pdf_content : str) :
        if(self.config['DataProcessing'] == 'Stemming'):
            os.makedirs("cache_s", exist_ok=True)
        else :
            os.makedirs("cache_l", exist_ok=True)
        with open(self.get_cache_path(pdf_file_name), 'w') as cache_file:
            cache_file.write(pdf_content)
        

    def populate_cache(self,pdf_file_name : str) -> None:
        pdf_content = ""
        pdf_path = os.path.join(self.folder_path, pdf_file_name)
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                pdf_content += page.extract_text()
        preprocessed_results = self.preprocess(pdf_content)   
        self.save_to_cache(pdf_file_name,preprocessed_results)
        

    def preprocess(self, pdf_content:str ) -> dict[str,str]:
        # Remove punctuation
        pdf_content = pdf_content.lower()
        pdf_content = re.sub(r'[^a-z\s]', '', pdf_content)

        # Remove stop words
        words = pdf_content.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        if(self.config["DataProcessing"] == "Stemming") :
            # Apply stemming
            ps = PorterStemmer()
            words = [ps.stem(word) for word in words]
        else :
            # Apply lemmatizing
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
        return  ' '.join(words)

    def process_texts(self) -> dict[str,str]:
        # check if already in cache
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        result = {}
        for pdf_file_name in pdf_files:
            if not self.has_cache(pdf_file_name):
                self.populate_cache(pdf_file_name)
            result[pdf_file_name] = self.load_from_cache(pdf_file_name)
        return result

