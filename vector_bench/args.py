from argparse import ArgumentParser

from vector_bench.dataset.source import DataSource


def build_arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--client",
        "-c",
        choices=["pgvecto_rs", "pgvector", "qdrant"],
        default="pgvecto_rs",
        help="client type",
    )
    parser.add_argument(
        "--worker-num",
        "-w",
        type=int,
        help="number of workers, if not set, use min(32, cpu_thread + 4)",
    )
    parser.add_argument("--url", "-u", help="database url")
    parser.add_argument(
        "--source", "-s", choices=DataSource.list(), help="dataset source"
    )
    parser.add_argument("--query", action="store_true", help="query benchmark")
    parser.add_argument("--insert", action="store_true", help="insert data")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)
