# Vector DB Internals (condensed)

General concepts (same in Chroma, FAISS, pgvector, Pinecone…), then where each lands
in our code.

## 1. Embeddings = geometry
A model maps text → a point in high-dim space (MiniLM = 384 floats). Trained so
**semantic similarity ≈ geometric closeness**. "Search" = find the document points
nearest the query point.

## 2. Measuring "close"
- **Dot product** `a·b = Σaᵢbᵢ` — mixes angle *and* length.
- **Euclidean (L2)** `‖a−b‖` — straight-line distance; smaller = closer.
- **Cosine similarity** `cos = (a·b)/(‖a‖‖b‖)` — the **angle only**, length divided out.
  Range −1…1 (1 = same dir, 0 = unrelated, −1 = opposite).

**Why cosine for text:** vector *magnitude* tracks incidental stuff (length, emphasis),
not meaning. Direction = "what it's about." So text retrieval uses cosine.

## 3. Normalization (the trick)
Scale every vector to length 1 (`â = a/‖a‖`). Then `‖a‖=‖b‖=1`, so:
```
cos(a,b) = a·b          # cosine collapses to a plain dot product
```
Pay the division **once** at embed time → every later comparison is a cheap dot product.
Bonus: on unit vectors, `‖a−b‖² = 2 − 2cos`, so **L2 and cosine give the same ranking** —
cosine semantics can ride on L2-optimized index machinery.

## 4. Distance vs similarity (bookkeeping)
Indexes minimize a **distance** (smaller = closer), but cosine sim is bigger = closer.
So DBs store **cosine distance = 1 − cosine similarity** (range 0…2). Convert back with
`sim = 1 − distance`.
⚠️ Convention varies per DB (FAISS raw inner product = bigger better; pgvector `<=>` =
distance). A silent sign-flip ranks results backwards — always check what your DB returns.

## 5. The search problem
Exact NN = compare query to all N vectors ("**flat**"/brute force), O(N·d).
- ~10⁴–10⁵ vectors: brute force is microseconds — optimal, no index needed.
- Millions: too slow → use **Approximate Nearest Neighbor (ANN)**: trade a little recall
  for 100–1000× speed. **The central vector-DB trade-off.**

This stacks **two** recall risks: model may not place the right doc near the query, *and*
the ANN index may not *find* a near doc that exists. Eval recall@k measures both at once.

## 6. HNSW (Hierarchical Navigable Small World)
The dominant ANN index (Chroma/Weaviate/Qdrant/pgvector). A **graph** you navigate:
- **Small-world graph:** each node links to ~M nearest neighbors; greedy-walk toward the
  query. Short + few long-range links ⇒ ~O(log N) hops.
- **Hierarchy (skip-list trick):** stacked layers — sparse "highway" on top, all nodes on
  the bottom. Search coarse→fine: navigate top, drop a layer, refine, repeat.

**Knobs:** `M` (links/node — recall↑, memory↑), `ef_construction` (build quality),
`ef_search` (query-time candidate list — the live **recall⇄latency dial**).

**Costs:** approximate (can miss); **whole graph in RAM** (memory is the real limit);
deletes are tombstoned/awkward until rebuild.

## 7. Other index families
- **Flat** — exact, no index; correct baseline to ~10⁵.
- **IVF** — k-means cells, search nearest few (`nprobe`); cheap memory, boundary misses.
- **PQ** — quantize/compress vectors to bytes; ~10–50× less memory, approximate. Often
  **IVF-PQ** for billion-scale (FAISS).
- **ScaNN / DiskANN** — SIMD- or SSD-resident variants.

Universal axes: **recall × latency × memory × build-time × update-friendliness.** Every
index is a different compromise. HNSW = great recall-at-low-latency, update-tolerant, costs RAM.

## 8. In our code
| Concept | File | Line |
|---|---|---|
| Normalize → dot = cosine | `embeddings.py` | `encode(..., normalize_embeddings=True)` |
| Cosine distance space | `vector_store.py` | `metadata={"hnsw:space": "cosine"}` |
| Distance → similarity | `vector_store.py` | `score = 1.0 - dists[i]` |
| ANN query (HNSW) | `vector_store.py` | `collection.query(...)` |
| HNSW overkill now | 262 chunks | brute force = µs; index pays off at 10⁵+ |
| Delete bookkeeping | `ingest.py` | `delete_by_source` |

**One-paragraph version:** an embedding turns meaning into a *direction*; compare
directions with *cosine*; *normalize once* so cosine = cheap dot product (and cosine/L2
rank alike); store as *distance* (`1 − cos`) so "nearest" = minimize; and since scanning
all N is too slow at scale, build an *approximate* index — usually **HNSW**, a
hierarchical small-world graph navigated coarse-to-fine — trading a little recall for big
speed, with `ef_search` as the dial.
