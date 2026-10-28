import string
from nltk.stem import PorterStemmer
import pickle
import os
import math
from math import log
from collections import defaultdict, Counter
from search_utils import (
    format_search_result,
    SearchResult,
    BM25_B,
    BM25_K1,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH, 
    load_movies, 
    CACHE_DIR,
    DOCMAP_PATH,
    INDEX_PATH,
    TF_PATH,
    DOC_LENGTHS_PATH)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}        

    def __add_document(self,doc_id:int,text:str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) ->float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]

    def get_idf(self,term:str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count + 1)/ (term_doc_count + 1))

    def get_tf_idf(self, doc_id:int,term:str):
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term:str)->float:
        N = len(self.docmap)
        df = len(self.index[term])
        idf = log((N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def get_bm25_tf(self, doc_id:int, term:str, k1=BM25_K1, b = BM25_B):
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id,term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self,doc_id:int, term:str) ->float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf*bm25_idf
    
    def bm25_search(self, query:str, limit = DEFAULT_SEARCH_LIMIT) ->list[SearchResult]:
        tokens = tokenize_text(query)
        scores = {}
        for id in self.docmap:
            bm25_score=0
            for token in tokens:
                bm25_score += self.bm25(id, token)
            scores[id] = bm25_score
        scores = sorted(scores.items(),key=lambda item: item[1], reverse = True)
        results:list[SearchResult] = []
    
        for doc_id, score in scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score
            )
            results.append(formatted_result)

        return results
        

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
        with open(DOC_LENGTHS_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(INDEX_PATH, "rb") as f:
            self.index=pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap=pickle.load(f)
        with open (TF_PATH, "rb") as f:
            self.term_frequencies=pickle.load(f)
        with open(DOC_LENGTHS_PATH, "rb") as f:
            self.doc_lengths=pickle.load(f)        

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

def bm_25_idf_command(term:str)->float:
    idx = InvertedIndex()
    idx.load()
    tokenized_term = tokenize_single_term(term)
    return idx.get_bm25_idf(tokenized_term)

def bm25_tf_command(doc_id:int, term:str, k1 = BM25_K1 or None, b = BM25_B or None):
    idx = InvertedIndex()
    idx.load()
    tokenized_term = tokenize_single_term(term)
    return idx.get_bm25_tf(doc_id,tokenized_term, k1, b)

def bm25search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)
