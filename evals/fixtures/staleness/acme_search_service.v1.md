# ACME Search Service — Configuration Reference

> Fixture document for the memory staleness eval (capability D). This is a deliberately
> FICTIONAL service whose facts do not appear anywhere else in the corpus, so the only way
> to answer questions about it is to retrieve THIS document. The eval harness ingests this
> v1 file for session 1, then swaps in `acme_search_service.v2.md` and re-ingests before
> session 2 — simulating a corpus that changed under the system. The file is NOT part of
> the standing corpus; it is added and removed by the staleness runner.

The ACME Search Service is the internal retrieval backend used by the Helix platform.

Embedding model: the service runs the `acme-embed-v1` embedding model to vectorize queries
and documents.

Default result count: by default the service returns the top 8 results for each query.

Index refresh interval: the index is rebuilt every 6 hours.
