
from PyPDF2 import PdfReader
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class DataProcessing :
    def __init__(self, config,folder_path) -> None:
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.folder_path = folder_path
        self.config = config
    
    def get_cache_path(self,pdf_file_name : str) -> str:
        file_path = ""
        
        if self.config.get('DataProcessing') == 'Stemming':
            file_path = os.path.join('cache_s', pdf_file_name)
        else :
            file_path = os.path.join('cache_l', pdf_file_name)
        return file_path
    
    def load_from_cache(self,pdf_file_name : str) -> str:
        with open(self.get_cache_path(pdf_file_name), 'r') as file:
            text = file.read()
            return text
        
    def extract_text_from_pdfs(self):
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        all_texts = {}
        for pdf_file_name in pdf_files:
            # check if the file is in cache
            pdf_content = ""
            if os.path.exists(self.get_cache_path(pdf_file_name)):
                pdf_content = self.load_from_cache(pdf_file_name)
            else : 
                print(f"Extracting text from {pdf_file_name}")
                pdf_path = os.path.join(self.folder_path, pdf_file_name)
                with open(pdf_path, 'rb') as file:
                    reader = PdfReader(file)
                    for page in reader.pages:
                        pdf_content += page.extract_text()
                        
                # store in cache
                if(self.config.get('DataProcessing') == 'Stemming'):
                    os.makedirs("cache_s", exist_ok=True)
                else :
                    os.makedirs("cache_l", exist_ok=True)
                with open(self.get_cache_path(pdf_file_name), 'w') as cache_file:
                    cache_file.write(pdf_content)
                all_texts[pdf_file_name] = pdf_content
        return all_texts

    def preprocess(self,papers : dict[str,str]) -> dict[str,str]:
        # Preprocess the text
        for name in papers :   
            text  = papers[name]
            text = text.lower()
            
            # Remove punctuation
            text = re.sub(r'[^a-z\s]', '', text)

            # Remove stop words
            words = text.split()
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
                papers[name] = ' '.join(words)
        return papers

    def process_texts(self) -> str:
        all_texts = self.extract_text_from_pdfs()
        return self.preprocess(all_texts)


