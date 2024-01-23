from vector_bench.args import build_arg_parser
from vector_bench.bench import Benchmark
from vector_bench.dataset.source import DataSource
from vector_bench.log import logger
from vector_bench.spec import DatabaseConfig


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    logger.info(args)

    source = DataSource.select(args.source)
    client_config = DatabaseConfig(
        vector_dim=source.vector_dim,
        distance=source.distance,
        table=args.source,
        url=args.url or None,
        name=args.client,
    )
    benchmark = Benchmark(client_config, source, worker_num=args.worker_num)
    if args.insert:
        benchmark.insert()
    if args.query:
        result = benchmark.query()
        result.display()
