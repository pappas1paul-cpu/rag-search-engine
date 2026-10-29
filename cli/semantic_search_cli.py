import argparse
from lib import semantic_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify language model and max sequence length")
    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying model and max sequence length")
            semantic_search.verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()