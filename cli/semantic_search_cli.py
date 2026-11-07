import argparse
import numpy
from lib import semantic_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify language model and max sequence length")
    search_parser = subparsers.add_parser("embed_text", help="Create text embeddings")
    search_parser.add_argument("text", type=str, help="Text to embed")
    verification_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    query_embedding_parser= subparsers.add_parser("embed_query", help = "Create query embeddings")
    query_embedding_parser.add_argument("query", type=str, help="User query for which to generate embeddings")
    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying model and max sequence length")
            semantic_search.SemanticSearch.verify_model()
        case "embed_text":
            print("embedding text...")
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            print("verifying embeddings...")
            semantic_search.verify_embeddings()
        case "embed_query":
            print("generating query embeddings")
            semantic_search.embed_query_text(args.query)
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()