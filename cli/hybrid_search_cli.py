import argparse
from lib import hybrid_search
from lib import search_utils

def main()-> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help ="Normalize BM_25 and Semantic Search results")
    normalize_parser.add_argument(
        "scores", nargs="*", type = float, help="Normalize a list of scores")
    weighted_parser = subparser.add_parser(
        "weighted-search", help="produces selection of keyword and semantic search"
    )
    weighted_parser.add_argument(
        "query", type=str, help="String to search for"
    )
    weighted_parser.add_argument(
        "--alpha", 
        type=float,
        default=search_utils.DEFAULT_ALPHA,
        help="weighing between keyword and semantic"
    )
    weighted_parser.add_argument(
        "--limit",
        type = int,
        default=search_utils.DEFAULT_SEARCH_LIMIT,
        help="How many results to show"
    )
    rrf_parser = subparser.add_parser(
        "rrf-search", help="Produce reciprocal rank fusion results"
    )
    rrf_parser.add_argument(
        "query", type=str, help="String to search for"
    )
    rrf_parser.add_argument(
    "-k", type=int, default=search_utils.RRF_K, help="RRF weight constant"
    )
    rrf_parser.add_argument(
        "--limit",type=int,default=search_utils.DEFAULT_SEARCH_LIMIT,help="How many results to show"
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = hybrid_search.normalize_scores(args.scores)
            for score in normalized_scores: 
                print(f"* {score:.4f}")
        case "weighted-search":
            result = hybrid_search.weighted_search_cmd(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case "rrf-search":
            result = hybrid_search.rrf_search_command(args.query, args.k, args.limit)
            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()

if __name__== "__main__":
    main()