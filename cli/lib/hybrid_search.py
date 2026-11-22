import os
from typing import TypedDict
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (SearchResult,
                           DEFAULT_ALPHA,
                           DEFAULT_SEARCH_LIMIT,
                           INDEX_PATH,
                           RRF_K,
                           format_search_result,
                           load_movies,
)
class WeightedSearchCommandResult(TypedDict):
    original_query:str
    query:str
    alpha:float
    results:list[SearchResult]

class CombinedScoreData(TypedDict):
    title:str
    document:str
    bm25_score: float
    semantic_score: float

class RRFScoreData(TypedDict):
    title: str
    document: str
    rrf_score: float
    bm25_rank: int | None
    semantic_rank: int | None

class RRFSearchCommandResult(TypedDict):
    original_query: str
    query: str
    k: int
    results: list[SearchResult]

class HybridSearch:
    def __init__(self, documents:list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query:str, limit:int)->list[dict]:
            self.idx.load()
            return self.idx.bm25_search(query,limit)
        
    def weighted_search(self, query:str, alpha:float, limit:int = 5
                           )->list[dict]:
            keyword_results = self._bm25_search(query, limit*500)
            semantic_results =self.semantic_search.search_chunks(query,limit*500 )
            combined = combine_search_results(keyword_results,semantic_results, alpha)
            return combined[:limit]
            
    def rrf_search(self,  query:str, k:int, limit:int=10)-> list[dict]:
        keyword_results = self._bm25_search(query, limit*500)
        semantic_results =self.semantic_search.search_chunks(query,limit*500 )
        combined = combine_rrf_results(keyword_results,semantic_results,k)
        return combined[:limit]
    
def normalize_scores(scores:list[float]) ->list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0]*len(scores)
    normalized_scores = []
    for score in scores:
        normalized_scores.append((score - min_score)/(max_score - min_score))
    return normalized_scores

def normalize_search_results(
    results: list[SearchResult],
) -> list[tuple[SearchResult, float]]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    return list(zip(results, normalized))


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1 / (k + rank)


def combine_search_results(bm25_results:list[SearchResult],
                           semantic_results:list[SearchResult],
                           alpha:float = DEFAULT_ALPHA,
                           ) ->list[SearchResult]:
    
    
    bm25_norm = normalize_search_results(bm25_results)
    semantic_norm = normalize_search_results(semantic_results)
    
    combined_scores: dict[int, CombinedScoreData] = {}

    for result, norm_score in bm25_norm:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title":result["title"],
                "document":result["document"],
                "bm25_score":0.0,
                "semantic_score":0.0,
            }
        combined_scores[doc_id]["bm25_score"] = max(
                combined_scores[doc_id]["bm25_score"], norm_score
        )
    for result, norm_score in semantic_norm:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title":result["title"],
                "document":result["document"],
                "bm25_score":0.0,
                "semantic_score":0.0,
                }
        combined_scores[doc_id]["semantic_score"] = max(
                combined_scores[doc_id]["semantic_score"], norm_score
                )
    hybrid_results: list[SearchResult] = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"],data["semantic_score"],alpha)   
        result = format_search_result(
            doc_id = doc_id,
            title=data["title"],
            document=data["document"],
            score = score_value,
            bm25_score = data["bm25_score"],
            semantic_score = data["semantic_score"]
        )    
        hybrid_results.append(result)
    return sorted(hybrid_results, key=lambda x:x["score"], reverse=True)

def combine_rrf_results(bm25_results:list[SearchResult],
                        semantic_results:list[SearchResult],
                        k:int = RRF_K)->list[SearchResult]:
    rrf_scores:dict[int,RRFScoreData] = {}
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id]={
                "title":result["title"],
                "document":result["document"],
                "rrf_score":0.0,
                "bm25_rank":None,
                "semantic_rank":None
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank,k)
    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id]={
                "title":result["title"],
                "document":result["document"],
                "rrf_score":0.0,
                "bm25_rank":None,
                "semantic_rank":None
            }
        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank,k)
    sorted_items = sorted(rrf_scores.items(), key=lambda x:x[1]["rrf_score"], reverse=True)
    rrf_results:list[SearchResult] = []
    for doc_id, data in sorted_items:
        result = format_search_result(
            doc_id = doc_id,
            title = data["title"],
            document = data["document"],
            score = data["rrf_score"],
            bm25_rank = data["bm25_rank"],
            semantic_rank = data["semantic_rank"]
        )
        rrf_results.append(result)
    return rrf_results
    
def weighted_search_cmd(query:str,alpha:float = DEFAULT_ALPHA, limit:int = DEFAULT_SEARCH_LIMIT) -> WeightedSearchCommandResult:
    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    results = searcher.weighted_search(query,alpha, limit)
    return {
        "original_query":original_query,
        "query":query,
        "alpha":alpha,
        "results":results
    }

def rrf_search_command(query:str, k:int = RRF_K, limit:int = DEFAULT_SEARCH_LIMIT) -> RRFSearchCommandResult:
    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    search_limit = limit
    results = searcher.rrf_search(query, k, search_limit)
    return {
        "original_query":original_query,
        "query":query,
        "k":k,
        "results":results
    }
