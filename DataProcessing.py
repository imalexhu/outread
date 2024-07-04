
from PyPDF2 import PdfReader
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class DataProcessing :
    def __init__(self,folder_path, config) -> None:
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.folder_path = folder_path
        self.config = config
    
    def extract_text_from_pdfs(self):
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        all_texts = {}

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.folder_path, pdf_file)
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = "Hello I am Alexander, and I am 8 years old. I feel good. I am happy"
                # for page in reader.pages:
                #     text += page.extract_text()
                # print(pdf_file)
                all_texts[pdf_file] = text
        return all_texts

    def preprocess(self,papers : dict[str,str]) :
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
        all_texts = self.extract_text_from_pdfs(self.folder_path)
        return self.preprocess(all_texts)


