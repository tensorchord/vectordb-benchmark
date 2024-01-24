# Vector DB Benchmark

Supported databases/extensions:

- [x] [`pgvecto.rs`](https://github.com/tensorchord/pgvecto.rs)
- [x] [`pgvector`](https://github.com/pgvector/pgvector)
- [ ] [`qdrant`](https://github.com/qdrant/qdrant/)

Supported datasets:

- [x] random generated
- [x] GIST 960

## Installation

```bash
pip install vector_bench
```

## Run

### Server

Run the docker compose file under [`server`](server/) folder.

```base
cd server/pgvecto.rs && docker compose up -d
```

### Client

```bash
# help
vector_bench --help
# only insert the data
vector_bench --insert --url postgresql://postgres:password@127.0.0.1:5432/postgres -s gist_960_l2
# only query the data (make sure the data is already inserted)
vector_bench --query --url postgresql://postgres:password@localhost:5432/postgres -s gist_960_l2
# insert and query the data
vector_bench --insert --query --url postgresql://postgres:password@localhost:5432/postgres -s gist_960_l2
```

## How to contribute

```bash
# install all the necessary dependencies:
make dev
# format code
make format
# lint
make lint
```

### Add more datasets

- Add new `DatasetConfig` to `vector_bench/dataset/source.py`

### Add more clients

- Inherit and implement the `BaseClient` class in `vector_bench/client/base.py`
