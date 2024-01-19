# Vector DB Benchmark

Supported databases/extensions:

- [x] [`pgvecto.rs`](https://github.com/tensorchord/pgvecto.rs)
- [ ] [`pgvector`]()

Supported datasets:

- [x] random generated
- [x] GIST 960


## Installation

```bash
pip install vector_bench[pgvectors]
```

## Run

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
