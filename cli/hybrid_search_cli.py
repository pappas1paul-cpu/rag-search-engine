import argparse
from lib import hybrid_search

def main()-> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help ="Normalize BM_25 and Semantic Search results")
    normalize_parser.add_argument(
        "scores", nargs="*", type = float, help="Normalize a list of scores")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = hybrid_search.normalize_scores(args.scores)
            for score in normalized_scores: 
                print(f"* {score:.4f}")
        case _:
            parser.print_help()

if __name__== "__main__":
    main()