# de-interview-wiki

Data Engineering interview knowledge base, built from normalized posts.

## Pipeline

The cloud pipeline is defined in
[`.github/workflows/run-python-worker.yml`](.github/workflows/run-python-worker.yml)
and runs in three stages:

1. **Discover** — scans `data/posts/*/post.json` and builds the matrix.
   No post list is hardcoded, so adding a post directory is enough to
   add a worker.
2. **Worker** — one matrix job per post. Each job creates a job
   manifest, runs the Python worker (`src/workers/cli.py`) to enrich the
   post through OpenCode + Space Bunny, and uploads its own artifact
   named `cloud-worker-result-<post_id>`. Workers never write to the
   repository and never push commits.
3. **Aggregate** — runs only after every worker succeeds. It downloads
   the worker artifacts, runs the Python aggregator, and uploads a
   single canonical `knowledge_base.json` as artifact
   `knowledge-base-<run_id>`.

## Aggregation

[`src/aggregation/aggregator.py`](src/aggregation/aggregator.py) merges
the per-worker JSON results into one canonical knowledge base.

```
python -m src.aggregation.aggregator \
  --input-dir aggregation/worker-results \
  --output aggregation/knowledge_base.json \
  --expected-post-count 3
```

Behavior:

- Worker job manifests are skipped. They are recognized structurally:
  a manifest has `job_id` and no `id`, a `KnowledgePost` is the
  opposite.
- Duplicate post IDs are a hard error, so a post is never silently
  overwritten.
- Invalid `KnowledgePost` payloads and unreadable files are hard errors.
- Posts are sorted by `id`, so the same inputs always produce the same
  ordering regardless of artifact download order.
- Output is written atomically via a `.tmp` file, so a failed run never
  leaves a partial knowledge base behind.
- `--expected-post-count` fails the run when a worker result is
  missing, instead of publishing a silently incomplete knowledge base.

Output payload:

```json
{
  "schema_version": 1,
  "generated_at": "...",
  "stats": {
    "result_files_found": 6,
    "posts_aggregated": 3,
    "files_skipped": 3
  },
  "skipped_files": ["...: worker job manifest"],
  "posts": []
}
```

## Tests

```
pip install -r requirements.txt
pytest
```

`tests/test_aggregator.py` covers aggregation, manifest skipping,
duplicate detection, invalid payloads, and deterministic ordering.

`tests/test_workflow_pipeline.py` validates the workflow structure
against the Python code, including an end-to-end check of the
downloaded artifact layout. It needs `PyYAML`, which is a test-only
dependency and is skipped automatically when unavailable:

```
pip install pyyaml
```

The other files under `tests/` are manual scripts that call the live
OpenCode CLI and are not part of automated collection.
