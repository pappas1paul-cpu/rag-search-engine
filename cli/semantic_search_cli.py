import argparse
from lib import semantic_search
from lib import search_utils

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    
    subparsers.add_parser(
        "verify", help="Verify language model and max sequence length")
    
    search_parser = subparsers.add_parser(
        "embed-text", help="Create text embeddings")
    search_parser.add_argument(
        "text", type=str, help="Text to embed")
    
    verification_parser = subparsers.add_parser(
        "verify-embeddings", help="Verify embeddings")
    
    query_embedding_parser= subparsers.add_parser(
        "embed-query", help = "Create query embeddings")
    query_embedding_parser.add_argument(
        "query", type=str, help="User query for which to generate embeddings")
    
    search_parser = subparsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    search_parser.add_argument(
        "query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=search_utils.DEFAULT_SEARCH_LIMIT, help="Number of results to return"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Splits long text into smaller pieces for embedding")
    chunk_parser.add_argument(
       "text", type=str, help="Text to chunk"
       )
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of each chunk in words"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default = 0, help = "Size of text overlap when chunking"
    )
    
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Splits long text into smaller semanticaly connected pieces"
    )
    semantic_chunk_parser.add_argument(
        "text", type=str, help="Text to semanticaly chunk"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Max chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default = 0, help="Word overlaping from chunk to chunk"
    )
    subparsers.add_parser(
        "embed-chunks", help="Generate embeddings for chunked documents"
    )
    search_chunked_parser = subparsers.add_parser(
        "search-chunked ", help="Search using chunked embeddings"
    )
    search_chunked_parser.add_argument(
        "query", type=str, help="Search query"
    )
    search_chunked_parser.add_argument(
        "--limit", type=int, help="Number of results to return"
    )

   
    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying model and max sequence length")
            semantic_search.SemanticSearch.verify_model()
        case "embed-text":
            print("embedding text...")
            semantic_search.embed_text(args.text)
        case "verify-embeddings":
            print("verifying embeddings...")
            semantic_search.verify_embeddings()
        case "embed-query":
            print("generating query embeddings")
            semantic_search.embed_query_text(args.query)
        case "search":
            semantic_search.semantic_search(args.query, args.limit)
        case "chunk":
            semantic_search.chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic-chunk":
            semantic_search.semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed-chunks":
            embeddings = semantic_search.embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search-chunked ":
            result = semantic_search.search_chunk_command(args.query, args.limit)
            print(f"Query: {result['query']}")
            print("Results:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()