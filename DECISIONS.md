# Caching Strategy Decisions

This document records design decisions, tradeoff measurements, and rejected approaches
for the semantic caching implementation. It is written after the fact, not as aspirational
design. Numbers are from production-like evaluation runs unless marked as estimated.

Last updated: 2026-03-29

---

## Decision Entry Format

Each decision entry uses the following structure:

- **Decision ID and title**
- **Status**: Chosen / Rejected / Superseded / Under Review
- **Context**: What problem were we solving, what constraints applied
- **Options considered**: Table of options with brief description
- **What we measured**: Actual numbers — latency (p50/p95), cost per 1000 queries, cache hit rate, quality delta. If estimated, say so.
- **Why the chosen option won**: The specific tradeoff that decided it
- **What was wrong about the first instinct**: The approach that seemed obvious and wasn't
- **Unresolved tension**: What this decision does NOT solve; what could cause us to revisit it

---

## DEC-001: Similarity Threshold Selection

**Status**: Chosen (with caveats)

**Context**: The single most consequential parameter in the entire system. Databricks Vector Search uses L2-distance-based similarity scoring: `1 / (1 + dist(q, x)²)`. This produces scores in a compressed range near zero for high-dimensional embeddings (1024 dims, GTE-large), not the 0.0–1.0 range you get with cosine similarity. Too low a threshold: the cache returns semantically different answers to different questions (quality regression). Too high: cache hit rate drops to near zero and you've added embedding latency with no benefit.

The EDA in `00_introduction.ipynb` cross-joins 100 synthetic questions (10 base questions × 10 paraphrases each) and plots the L2-score distribution for similar vs. dissimilar pairs. The distributions overlap — there is no clean separating threshold.

**Options considered**:

| Threshold | Recall (similar pairs) | Precision (approx.) | Quality Risk |
|-----------|----------------------|---------------------|--------------|
| 0.005     | ~75%+                | Low — many false positives | High: dissimilar questions return cached answers |
| 0.01      | ~50% (estimated)     | Moderate            | Moderate: some false positives in overlap zone |
| 0.015     | ~25%                 | High                | Low: few false positives but most cache entries go unused |

**What the evaluation notebook measured** (`04_evaluate.ipynb`):
- Answer relevance (LLM-as-judge, Llama 3.1 70B): standard RAG scored **4.63/5**, cached chain scored **4.53/5** — a delta of **-0.10** on a 5-point scale.
- The cached chain ran **>2x faster** than the standard chain on the same 100-question eval set.
- Cache hit detection: traced via span count (6 spans = cache hit, more = full RAG execution).

**First instinct**: Start at 0.015 — "conservative" felt safe. But at 0.015, recall drops to ~25%, meaning three out of four semantically identical paraphrases miss the cache. The system adds ~20ms of embedding latency on every request for negligible savings.

**What we actually learned**: The L2-score distribution is dataset-dependent. The overlap zone between similar and dissimilar pairs means any single global threshold is a compromise. The current implementation uses a single global threshold of `0.01` (`config.py`, `SIMILARITY_THRESHOLD`). This was chosen as a "balanced starting point" per the notebook — it is not the result of per-category optimization.

**Chosen approach**: Global threshold of 0.01 for the demo workload. For production deployments, threshold must be calibrated per query category: factual documentation Q&A can tolerate lower thresholds (higher recall), analytical or generative queries need higher thresholds. A single global threshold is the wrong abstraction long-term.

**Unresolved tension**: Threshold calibration requires a labeled eval set with ground-truth semantic groupings. For new deployments without query history, the threshold is a guess. The synthetic question generation technique used here (paraphrase 10 base questions) does not represent real query distributions. Cold-start threshold selection remains unsolved.

---

## DEC-002: Cache Store — Direct-Access Vector Search vs. Delta Sync Index vs. External Store

**Status**: Chosen

**Context**: The cache needs to store (query embedding, response text) pairs and support ANN search at low latency. The write path must handle both bulk warm-up and per-request lazy population.

**Options considered**:

| Option | ANN Latency (est.) | Write Behavior | Unity Catalog Governance | Ops Overhead |
|--------|-------------------|----------------|-------------------------|--------------|
| Databricks VS — Direct-Access Index | ~40-80ms (estimated) | Synchronous upsert | Native (index is UC-governed) | None (managed) |
| Databricks VS — Delta Sync Index | ~40-80ms (estimated) | Async via CDF, 5-30s lag | Native | Low |
| Redis (ElastiCache, HNSW) | ~5-15ms (estimated) | Synchronous | None | Moderate |
| Postgres + pgvector | ~20-60ms (estimated) | Synchronous | None | High |

**First instinct**: Delta Sync index. It is the default recommendation in most Databricks Vector Search documentation and handles the Delta-to-index pipeline automatically. It was a reasonable starting point because it means the "source of truth" is a Delta table, which is familiar and auditable.

**Why we didn't choose it**: The cache is not derived from an existing Delta table — it is populated dynamically at runtime via upserts (both batch warm-up in `warm_cache()` and per-request writes in `store_in_cache()`). A Delta Sync index would require writing to a Delta table first, then waiting for CDF propagation (5-30s) before the entry becomes searchable. That lag means a cache-miss response stored on request N would not be available as a cache hit for request N+1 if it arrives within the sync window. For a cache, this defeats the purpose.

**Why not Redis**: The cache table is the source of truth for cache quality analysis, hit rate reporting, and MLflow evaluation. With Redis, that data lives outside Unity Catalog, outside lineage tracking, and is inaccessible to SQL analytics. Every time we needed to understand why the cache returned a bad answer, we would have to reconstruct the query from application logs. That debugging tax outweighs the 30-40ms latency advantage.

**Chosen**: Direct-access Vector Search index. The implementation in `utils.py` calls `create_direct_access_index()` with a schema defined in `config.py`. Upserts are synchronous — `vs_index_cache.upsert([document])` in `cache.py` — so entries are immediately searchable. No CDF lag.

**Tradeoff accepted**: Direct-access indexes do not have a backing Delta table that can be queried with SQL independently. The index IS the store. This makes ad-hoc analysis of cache contents harder — you cannot just `SELECT * FROM cache_table`. Debugging requires querying through the Vector Search API.

**Unresolved tension**: Direct-access indexes have undocumented scaling limits. We have not stress-tested beyond ~50K entries. If the index becomes the bottleneck, migrating to a Delta Sync index means accepting the CDF lag or redesigning the write path.

---

## DEC-003: Embedding Model Selection for Cache Keys

**Status**: Chosen

**Context**: The embedding model used to generate cache keys adds latency to every single request (both hits and misses). It must be fast and semantically stable — the same question must produce the same or nearly identical vector reliably. The embedding used for cache keys does not need to match the embedding used for RAG document retrieval; these are separate concerns.

**Options considered**:

| Model | Dims | Cache Key Latency p50 (est.) | Semantic Stability | Cost (est.) |
|-------|------|-------------------------------|-------------------|-------------|
| text-embedding-ada-002 (OpenAI) | 1536 | ~40ms (API round-trip, external) | High | ~$0.10/1M tokens |
| GTE-large-en (Databricks hosted) | 1024 | ~20ms (estimated) | High | ~$0.04/1M tokens (DBU est.) |
| BGE-small-en-v1.5 (Databricks hosted) | 384 | ~12ms (estimated) | Medium | ~$0.01/1M tokens (DBU est.) |
| all-MiniLM-L6-v2 (local) | 384 | ~5ms (estimated) | Low | Near zero |

**First instinct**: Use the same embedding model as the RAG retrieval pipeline for "consistency." This was correct in principle and wrong in practice when the retrieval model is an external API (e.g., ada-002). Every cache lookup would add an external API round-trip to the hot path, creating a hard dependency on every request, hit or miss.

**Chosen**: `databricks-gte-large-en` on Databricks Model Serving (`config.py`, `EMBEDDING_MODEL_SERVING_ENDPOINT_NAME`). The embedding call stays within the Databricks network boundary — no external API dependency on the critical path. GTE-large produces 1024-dimensional vectors with high semantic stability for English text.

**Tradeoff accepted**: Cache keys use a different embedding space than RAG retrieval documents (if the retrieval pipeline uses a different model). Exact embedding comparisons across the two systems are invalid. This is acceptable because the cache and the retrieval index are separate artifacts with separate purposes — the cache matches questions to questions, not questions to document chunks.

**Unresolved tension**: Model Serving cold start means the embedding endpoint can spike to 400-600ms on the first call after idle. The current deployment configuration sets `scale_to_zero_enabled: true` (`utils.py`, `deploy_model_serving_endpoint()`), which means cold starts are expected. There is no keep-warm ping on the embedding endpoint.

---

## DEC-004: Cache Population Strategy — Eager vs. Lazy vs. Hybrid

**Status**: Chosen

**Context**: When and how cache entries are created determines the cold-start experience and the steady-state hit rate.

**Options considered**:

| Strategy | Cold-Start Hit Rate | Steady-State Hit Rate | Implementation Cost |
|----------|--------------------|-----------------------|---------------------|
| Lazy (write-on-miss) | 0% | Builds over time | Low — `store_in_cache()` on every miss |
| Eager (pre-seeded) | Depends on seed quality | Same as lazy after convergence | High — requires representative corpus |
| Hybrid | Moderate | Same | Medium |

**What we measured** (estimated from the eval workload): With the seed corpus of synthetic Q&A pairs loaded via `warm_cache()` (`data/synthetic_qa.txt`), the cache hit rate on the 100-question eval set was meaningfully higher than without seeding. Without the seed corpus (pure lazy), there are zero cache hits until real traffic starts populating entries. By week 3 of live traffic, both approaches converge to ~30-40% (estimated) as the cache fills from real queries.

**First instinct**: Eager seeding — intuitive to want the cache warm before go-live. The implementation cost was higher than expected: generating a representative seed corpus requires either (a) query log history you don't have yet, or (b) LLM-generated synthetic questions, which introduces a question quality dependency. The current seed set in `data/synthetic_qa.txt` was generated by LLM and covers only Databricks ML documentation topics.

**Chosen**: Hybrid. Pre-seed with synthetic Q&A pairs during deployment (`warm_cache()` in `cache.py`), then lazy-populate from live traffic (`store_in_cache()` called inside `call_model()` in `chain/chain_cache.py`). Every cache miss runs the full RAG chain and writes the result back into the cache.

**How the code implements it**: `chain_cache.py` line 78-84 — `call_model()` wraps the LLM invocation and unconditionally calls `semantic_cache.store_in_cache()` with the question and response. There is no quality gate — every response gets cached regardless of quality.

**Unresolved tension**: Cache entries from early traffic may be lower quality (early users are often developers testing with adversarial or nonsensical inputs). A bad RAG response on day 1 gets cached and served to semantically similar future queries. There is no quality gate on cache population. This is a known poison vector.

---

## DEC-005: Cache Eviction Strategy

**Status**: Chosen (FIFO and LRU implemented; TTL not implemented)

**Context**: Cache entries can become stale if the underlying documentation or data source changes. Without eviction, the cache serves outdated answers indefinitely.

**Options considered**:

| Strategy | Stale Content Risk | Ops Complexity | Hit Rate Impact |
|----------|-------------------|----------------|-----------------|
| No TTL (manual eviction only) | High | Low | None |
| Global TTL (e.g., 7 days) | Medium | Low | ~15-20% drop after each eviction cycle (estimated) |
| FIFO (oldest entries first) | Medium | Low | Evicts by insertion order, not usefulness |
| LRU (least recently used) | Lower | Medium | Retains frequently-hit entries |
| Quality-score-based eviction | Low | Very High | Unknown |

**First instinct**: Global TTL of 7 days — simple and predictable. For documentation Q&A where the source changes infrequently (monthly releases), a 7-day TTL is far too aggressive. It discards good cache entries with high hit rates every week and forces cold-start cost again.

**Chosen**: FIFO and LRU eviction strategies, triggered explicitly (not on a timer). Both are implemented in `cache.py` (`_evict_fifo()`, `_evict_lru()`). Eviction is invoked manually: `semantic_cache.evict(strategy='FIFO', max_documents=N)`. There is no automatic TTL, no scheduled eviction, and no source-change-triggered invalidation.

**Implementation detail worth noting**: The FIFO strategy sorts by `created_at`. The LRU strategy references a `last_accessed` column — but **this column does not exist in the index schema** defined in `config.py`. The schema defines: `id`, `creator`, `question`, `answer`, `access_level`, `created_at`, `text_vector`. There is no `last_accessed` field, and `get_from_cache()` does not update any access timestamp on cache hits. LRU eviction will fail or produce incorrect results in the current implementation. This is a known gap.

**Eviction mechanics**: Both strategies use a zero-vector similarity search (`[0] * 1024`) to retrieve candidate entries for deletion, then delete by ID in batches. This is a workaround for the lack of a `scan` or `list_all` API on direct-access indexes. It is not guaranteed to return the oldest or least-recently-used entries — it returns entries closest to the zero vector, which is arbitrary.

**Unresolved tensions**:
- LRU requires schema and write-path changes that have not been implemented.
- Source-lineage-based eviction (invalidate cache entries when the underlying document changes) is not implemented. The cache schema has no field linking a cached answer to the source document chunk it was derived from. Lineage tracking should have been in the schema from day one.
- The zero-vector scan trick for eviction is O(n) and single-threaded. The `05_cache_eviction.ipynb` notebook documents this limitation and suggests parallelism and bulk deletion as improvements.

---

## DEC-006: Routing Architecture — LangChain Runnable vs. MLflow Pyfunc

**Status**: Chosen

**Context**: The routing layer decides whether a query gets served from cache or runs the full RAG pipeline. Two implementation options were available within the Databricks ecosystem.

**Options considered**:

| Option | Composition Model | Tracing | Deployment | Flexibility |
|--------|------------------|---------|------------|-------------|
| LangChain Runnable chain | `RunnableLambda` + `itemgetter` | MLflow autolog (`mlflow.langchain.autolog()`) | `mlflow.langchain.log_model()` → Model Serving | High — composable |
| MLflow Pyfunc | Custom `predict()` method | Manual `mlflow.start_span()` | `mlflow.pyfunc.log_model()` → Model Serving | Full control |

**First instinct**: MLflow Pyfunc. It gives full control over the predict method, error handling, and logging. The routing logic (cache hit → return cached answer, cache miss → run RAG → store result) is straightforward procedural code that doesn't naturally fit a chain abstraction.

**Why we chose LangChain Runnable**: MLflow's `langchain.autolog()` provides automatic tracing of every step in the chain — each `RunnableLambda` becomes a traced span. This is critical for the evaluation workflow: the `04_evaluate.ipynb` notebook uses span count (6 spans = cache hit path, more = full RAG path) to determine cache hit rate from inference table logs. With Pyfunc, tracing each step would require manual instrumentation.

**How the code implements it** (`chain/chain_cache.py`):
```
full_chain = (
    itemgetter("messages")
    | RunnableLambda(extract_user_query_string)
    | RunnableLambda(semantic_cache.get_from_cache)
    | RunnableLambda(router)
    | StrOutputParser()
)
```
The `router()` function (line 91-95) checks if `qa["answer"]` is empty string. If empty → cache miss, route to `rag_chain`. If populated → cache hit, return the cached answer directly.

**Tradeoff accepted**: The chain composition makes the routing logic less readable than a simple if/else in a Pyfunc `predict()` method. The `router()` function returns either a Runnable (the rag_chain) or a plain string, relying on LangChain's type coercion to handle both. This is not obvious to someone reading the code for the first time.

**Unresolved tension**: The span-count heuristic for detecting cache hits is fragile. If the chain structure changes (add a step, rename a lambda), the span count changes and the evaluation logic in `04_evaluate.ipynb` breaks silently. There is no explicit `cache_hit: true/false` flag logged in the trace metadata.

---

## DEC-007: Quality Measurement — What "Good" Means for a Cache Hit

**Status**: Chosen

**Context**: A cache hit is only valuable if the returned answer is correct for the new query. The evaluation framework uses MLflow `evaluate()` with LLM-as-a-judge (Llama 3.1 70B Instruct) to assess answer quality.

**What we measured** (`04_evaluate.ipynb`, eval set of 100 synthetic questions with ground-truth answers):

| Metric | Standard RAG | Cached Chain | Delta |
|--------|-------------|--------------|-------|
| Answer relevance (LLM judge, 0-5) | 4.63 avg | 4.53 avg | -0.10 |
| Total execution time (100 queries) | Baseline | >2x faster | Significant |

Additional metrics collected but not reported as aggregates in the notebook: `answer_similarity`, `answer_correctness` (both via LLM-as-judge).

**Interpretation**: The -0.10 relevance delta on a 5-point scale is small. For a documentation chatbot, this is acceptable. It would NOT be acceptable for a regulated workload (clinical, legal, financial) where answer quality is part of an SLA.

**First instinct**: Optimize for cost reduction — if the cache saves money, ship it. The measured quality delta made us reconsider: for any workload where answer quality is contractual, the threshold needs to be calibrated against quality, not just hit rate.

**How evaluation works**: The 100 questions are 10 base questions × 10 paraphrases. Ground truth answers come from `data/synthetic_qa.txt` (the same data used to warm the cache). This means the evaluation is partially testing whether the cache returns its own seed data — not whether it generalizes to unseen queries. The evaluation is necessary but not sufficient.

**Unresolved tensions**:
- The LLM judge (Llama 3.1 70B) is itself subject to variability. We do not have a human-labeled ground truth set beyond the synthetic 100 questions. At scale, we don't actually know if our quality measurements are accurate — we know they are consistent.
- Quality evaluation is a batch process run during development. There is no continuous quality monitoring of cache hits in production. A cache entry that was correct at write time can become factually outdated without triggering any alert.
- The evaluation does not measure answer freshness — a cached answer about a feature that changed last week may score high on relevance but be factually wrong.

---

## DEC-008: Caching Granularity in Agentic Workflows

**Status**: Under Review

**Context**: In a Mosaic AI Supervisor multi-agent system, caching can be applied at multiple levels: (a) top-level user query, (b) subagent tool call, or (c) both. This decision is not yet implemented in the current codebase but is part of the target architecture for production deployments.

**Options considered**:

| Level | Hit Rate Potential | Correctness Risk | Latency Savings |
|-------|-------------------|------------------|-----------------|
| Top-level user query only | Low (~12% estimated in agentic workloads) | Low | Moderate |
| Subagent tool calls only | High (~55% estimated for read-only tools) | Medium | High |
| Both | Highest | High (stale tool results) | Highest |

**First instinct**: Top-level only — seemed safest. In an agentic customer support context, top-level user queries are highly variable (billing, account changes, plan upgrades all look different on the surface). Hit rate of ~12% (estimated) means the cache is nearly irrelevant.

**What worked in practice** (estimated, from EchoStar/Boost Mobile workload context): Caching at the subagent tool call level. The "retrieve customer account details" tool is called with the same account_id across many different top-level conversations. Hit rate at tool call level was ~55% (estimated) for read-only data tools, with a 24-hour TTL.

**Tradeoff accepted**: Tool call caching introduces correctness risk for any tool that accesses mutable state. Caching should be applied ONLY to read-only tools with an explicit `cacheable: true` annotation in the tool schema. Mutating tools (update, create, delete) must never be cached.

**Unresolved tension**: The `cacheable` annotation is a convention, not enforced by any framework. A developer adding a new tool with side effects who forgets to set `cacheable: false` would silently cache mutating operations. There is no static analysis, runtime guard, or test coverage for this. It is a footgun.

---

## Open Questions

- **Index scaling limits**: At what entry count does the direct-access Vector Search index degrade under continuous cache writes? We have not stress-tested above ~50K entries.
- **Multi-tenant cache isolation**: Currently all users share the same cache. The schema includes `access_level` and `creator` fields, but `get_from_cache()` does not filter on them. For a multi-tenant system where User A's question about Account X should not return a cached result for User B, this is a correctness issue, not a feature request.
- **Cache hit rate drift**: How do we measure cache hit rate drift over time as the query distribution shifts? The current setup does not log whether a served answer was user-rated as helpful. Hit rate is derived from span count in inference table traces, not from an explicit metric.
- **Embedding cold-start detection**: What is the right behavior when the embedding endpoint is cold and cache lookup latency spikes to 400-600ms? Currently the system does not detect this and will return a cache hit slower than a direct LLM call would have been. There is no circuit breaker.
- **Freshness vs. relevance**: The quality evaluation uses answer relevance as the primary metric. It does not measure answer freshness. A cached answer about a product feature that changed last week may score high on relevance but be factually outdated.
- **Eviction correctness**: The FIFO/LRU eviction uses zero-vector similarity search to find candidates. This does not guarantee ordering by `created_at` or `last_accessed` — it returns entries nearest to the zero vector, which is semantically arbitrary. The eviction may remove high-value entries while retaining stale ones.
- **Evaluation circularity**: The cache is warmed with `data/synthetic_qa.txt`, and the evaluation ground truth is also derived from the same file. The evaluation partially measures whether the cache can return its own seed data, which inflates reported hit rates and quality scores relative to real-world performance.

---

## What We Would Change If Starting Again

- **Lineage tracking in the schema from day one.** The cache entry schema should include a reference to the source document chunk(s) that produced the cached answer. Without this, source-change-triggered eviction is impossible. Retrofitting lineage into an existing cache requires re-embedding and re-populating every entry.

- **The `last_accessed` field should be in the schema and updated on every cache hit.** LRU eviction is implemented in code but cannot work without this field. The schema was designed for FIFO only; LRU was added later without updating the data model.

- **Per-category similarity thresholds should have been the starting design.** A single global threshold (`SIMILARITY_THRESHOLD = 0.01`) is a compromise that is wrong for every query category in a different direction. The architecture should support threshold lookup by query classification from the start.

- **An explicit `cache_hit` flag in trace metadata**, not span-count inference. The current evaluation logic in `04_evaluate.ipynb` counts spans to determine cache hits — if the chain structure changes, the detection breaks silently.

- **Quality gate on cache population.** Not every RAG response deserves to be cached. A confidence score or answer-relevance check before writing to cache would prevent poisoning from low-quality early responses. The cost is one additional LLM-as-judge call per cache miss, which may not be acceptable — but the tradeoff should be made explicitly.

- **The embedding endpoint should not scale to zero** for the cache use case. Cold-start latency on the embedding model (400-600ms estimated) is added to every request. A keep-warm configuration or minimum replica count should be part of the deployment runbook, not an afterthought.

- **The `access_level` and `creator` fields in the schema are unused.** `get_from_cache()` does not filter on either field. If multi-tenant isolation is a requirement, the filtering must be added to the similarity search call. If it is not a requirement, the fields add schema complexity for no benefit and should be removed.
