import nltk
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from langchain_community.document_loaders import PyPDFLoader
from collections import defaultdict
from typing import Union, List
from langchain.text_splitter import CharacterTextSplitter
import cohere

class Agent:
    def __init__(self, api_key):
        self.client = cohere.ClientV2(api_key)
        self.embedding_model = "embed-english-v3.0"
        self.text_model = "command-r-plus-08-2024"
        self.rerank_model = "rerank-v3.5"
        self.generating_model = "command-r"
        self.input_document = "search_document"
        self.input_query = "search_query"
        self.indices = 10
        self.splits = None
        self.vectordb = None

    def load_paper(self, file_path, embed=False) -> Union[List[str], str]:
        document = PyPDFLoader(
            file_path=file_path
        ).load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.splits = text_splitter.split_documents(document)
        if embed:
            return [split.page_content for split in self.splits]
        else:
            return " ".join([split.page_content for split in self.splits])

    def document_embeddings(self, file_path) -> list:
        self.doc_embeds = self.client.embed(
            texts=self.load_paper(file_path, embed=True),
            model=self.embedding_model,
            input_type=self.input_document,
            embedding_types=["float"]
        ).embeddings.float
        self.vectordb = defaultdict(np.array, {
            i: np.array(embedding) for i, embedding in enumerate(self.doc_embeds)
        })
        return self.doc_embeds

    def query_embeddings(self, query) -> list:
        self.query_embeds = self.client.embed(
            texts=[query],
            model=self.embedding_model,
            input_type=self.input_query,
            embedding_types=["float"]
        ).embeddings.float[0]
        return self.query_embeds

    def rag(self, file_path, query) -> str:
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        document_embeds = self.document_embeddings(file_path)
        query_embeds = self.query_embeddings(query)

        similarities = [cosine_similarity(query_embeds, embedding) for embedding in document_embeds]
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[:self.indices]
        top_chunks_after_retrieval = [self.splits[int(i)].page_content for i in top_indices]

        rerank_response = self.client.rerank(
            query=query,
            documents=top_chunks_after_retrieval,
            top_n=3,
            model=self.rerank_model
        )

        indices = [result.index for result in rerank_response.results]
        top_chunks_after_rerank = [top_chunks_after_retrieval[i] for i in indices]

        preamble = """
        ## Task & Context
        You help people answer their questions and other requests interactively.
        You will be asked questions related to a particular modern topic.
        You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer.
        You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
        """

        documents = [
            {"title": i,
             "snippet": top_chunks_after_rerank[i],
             "data": {"text": top_chunks_after_rerank[i]}} for i in range(min(len(top_chunks_after_rerank), self.indices))
        ]

        response = self.client.chat(
            messages=[{"role": "user", "content": f"{preamble} {query}"}],
            documents=documents,
            model=self.generating_model,
            temperature=0.5,
        )
        return response.message.content[0].text

# Preprocessing functions
def preprocess_text(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    return " ".join(lemmatized_tokens)

# Feature extraction
def extract_features_tfidf(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

def extract_features_word2vec(texts):
    tokenized_texts = [text.split() for text in texts]
    return Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

def extract_features_bert(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return [tokenizer(text, return_tensors='pt', padding=True, truncation=True) for text in texts]
