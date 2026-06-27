# ACME Search Service — Configuration Reference

> Fixture document for the memory staleness eval (capability D), VERSION 2. Same document
> as `acme_search_service.v1.md` but with three values CHANGED (embedding model, default
> result count, index refresh interval). The harness swaps this in and re-ingests before
> session 2, so a system with stale memory will serve the v1 answers it cached in session 1
> instead of these current v2 values. The file is NOT part of the standing corpus.

The ACME Search Service is the internal retrieval backend used by the Helix platform.

Embedding model: the service runs the `acme-embed-v2` embedding model to vectorize queries
and documents.

Default result count: by default the service returns the top 15 results for each query.

Index refresh interval: the index is rebuilt every 2 hours.
