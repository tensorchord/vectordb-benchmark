from vector_bench.args import build_arg_parser
from vector_bench.bench import Benchmark
from vector_bench.dataset.source import DataSource
from vector_bench.log import logger
from vector_bench.spec import DatabaseConfig


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    logger.info(parser)

    source = DataSource.select(args.source)
    client_config = DatabaseConfig(
        vector_dim=source.vector_dim, url=args.url, name=args.client
    )
    benchmark = Benchmark(client_config, source)
    if args.insert:
        benchmark.insert()
    if args.query:
        benchmark.run()
