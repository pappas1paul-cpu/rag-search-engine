import string
from nltk.stem import PorterStemmer
import pickle
import os
import math
from collections import defaultdict, Counter
from search_utils import (
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH, 
    load_movies, 
    CACHE_DIR,
    DOCMAP_PATH,
    INDEX_PATH,
    TF_PATH)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)        

    def __add_document(self,doc_id:int,text:str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]

    def get_idf(self,term:str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count + 1)/ (term_doc_count + 1))

    def get_tf_idf(self, doc_id:int,term:str):
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_documents(self, term:str) -> list[int]:
        doc_ids = []
        for id in self.index.get(term, set()):
            doc_ids.append(id)
        return sorted(doc_ids)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie["description"]}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open(INDEX_PATH, "rb") as f:
            self.index=pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap=pickle.load(f)
        with open (TF_PATH, "rb") as f:
            self.term_frequencies=pickle.load(f)
        

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found, exiting...")
        exit()
    query_tokens=tokenize_text(query)
    results, seen = [], set()
    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results
    

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def tokenize_single_term(term:str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    return tokens[0]

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, tokenize_single_term(term))

def idf_command(term:str) ->float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(tokenize_single_term(term))

def tf_idf_command(doc_id:int, term:str) ->float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)