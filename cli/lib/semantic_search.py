from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from typing import Any
import numpy as np
import os
import json
from pathlib import Path
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    MOVIE_EMBEDDINGS_PATH,
    Movie,
    SemanticSearchResult,
    load_movies)

EmbeddingArray = NDArray[Any]
def semantic_search(query:str, limit:int = DEFAULT_SEARCH_LIMIT) -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)    

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

def embed_query_text(query:str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def search(self, query:str, limit:int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        similarities: list[tuple[float, Movie]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding,doc_embedding)
            similarities.append((similarity, self.documents[i]))
        similarities.sort(key = lambda x:x[0], reverse=True)
        results: list[SemanticSearchResult] = []
        for score, doc in similarities:
            results.append(
                {
                    "score":score,
                    "title":doc["title"],
                    "description":doc["description"]
                }
            )
        return results
        
    def build_embeddings(self, documents:list[dict])->EmbeddingArray:
        self.documents=documents
        movies = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            movies.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar = True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents:list[dict])->EmbeddingArray:
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

    def generate_embedding(self, text:str) ->EmbeddingArray:
        if not text:
            raise ValueError("No text provided")
        text = [text]
        embedded_text = self.model.encode(text)
        return embedded_text[0]
