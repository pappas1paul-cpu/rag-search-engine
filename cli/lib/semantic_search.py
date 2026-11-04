from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from pathlib import Path

def verify_embeddings():
    search = SemanticSearch()
    movies = []
    script_dir = Path(__file__).resolve().parent.parent.parent
    with open(script_dir/"data"/"movies.json", 'r',encoding='utf-8') as f:
        movies = json.load(f)
        movies = movies["movies"]
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
    f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
)

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents:list[dict]):
        self.documents=documents
        movies = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            movies.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar = True)
        np.save(r"cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents:list[dict]):
        self.documents=documents
        movies = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            movies.append(f"{document['title']}: {document['description']}")
        if os.path.exists(r"cache/movie_embeddings.npy"):
            self.embeddings = np.load(r"cache/movie_embeddings.npy")
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)
            

    def verify_model(self):
        search_instance = SemanticSearch()
        print(f"Model loaded: {search_instance.model}")
        print (f"Max sequence length: {search_instance.model.max_seq_length}")

    def generate_embedding(self, text):
        if not text:
            raise ValueError("No text provided")
        text = [text]
        embedded_text = self.model.encode(text)
        return embedded_text[0]
