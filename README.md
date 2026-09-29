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

`tests/test_wiki_generator.py` covers the static wiki generator.

`tests/test_agent_control_plane.py` covers the remote OpenCode control
plane: actor authorization, trigger conventions, task extraction,
prompt construction, credential redaction, validation-run resolution
and report rendering.

## Remote OpenCode Control Plane

You can drive development on this repository from a phone. A GitHub
issue acts as the command channel, and GitHub Actions does the work:

```
ChatGPT or Android
  -> GitHub issue titled "[OpenCode] <task>"
  -> OpenCode Remote Agent workflow
  -> OpenCode inspects, implements, tests, commits, pushes
  -> Run Python Workers validates
  -> repair cycle if it fails, bounded at 3
  -> result reported back into the issue
```

Your PC does not need to be running. Everything happens on GitHub's
runners.

### Starting a task

Create an issue whose title begins with `[OpenCode]` and put the full
task in the body.

```
Title:  [OpenCode] Add a retry to the aggregator
Body:   The aggregator should retry a worker result file that fails
        to parse. Add a test. Do not change the output format.
```

The workflow starts automatically on issue creation.

### Continuing a task

Comment on the same issue with an explicit command:

```
/continue also handle the missing-file case
```

`/opencode` is accepted as a synonym. The agent receives the original
task, your latest instruction, recent issue comments, and the current
repository state. It does not receive the full issue history; only the
most recent comments are included, each length-bounded.

### Supported commands

| Entry mode | Trigger | Convention |
| --- | --- | --- |
| New issue | `issues: opened` | Title starts with `[OpenCode]` |
| Continuation | `issue_comment: created` | Body starts with `/continue` or `/opencode` |
| Manual | `workflow_dispatch` | `task` input, optional `issue_number` |

Comments that do not start with a command are ignored. Issues without
the prefix are ignored. An ignored event exits without running the
agent and records why, visible as a short artifact.

### Manual dispatch

Useful for testing or emergencies, from the Actions tab:

- **Run Python Workers** → the main validation pipeline
- **OpenCode Remote Agent** → an ad-hoc agent task, with `task`,
  `issue_number` and `branch` inputs

A dispatch with no `issue_number` reports through the job summary and
an artifact instead of an issue comment.

### Results and reports

Each task ends with one comment containing a structured report:

```
## OpenCode Task Report

Status: SUCCESS | BLOCKED | BLOCKED_AFTER_3_ATTEMPTS | FAILED
Task / Commit / Tests / Validation workflow / CI result
Recovery attempts: n/3
Files changed / Final result / Human action required
```

The same information is written to the GitHub Actions job summary.
Full OpenCode output, the exact prompt, the validation logs and the
report are kept as workflow artifacts:

- `opencode-agent-logs-<run_id>` — stdout, stderr and prompt, 14 days
- `opencode-report-<run_id>` — report and failure evidence, 30 days

### Retry behaviour

The agent triggers `run-python-worker.yml` for the commit it pushed,
waits for that specific run, and repairs failures. It resolves the run
by commit SHA, so a concurrent run is never mistaken for its own.

At most **3** repair cycles. After that the report says
`BLOCKED_AFTER_3_ATTEMPTS` and includes the failure evidence. The
agent never reports success it did not achieve.

### Concurrency

One writer at a time per issue:

```
concurrency:
  group: opencode-agent-<issue number>
  cancel-in-progress: false
```

Tasks for the same issue are serialized. A queued task waits rather
than cancelling the running one, so an in-flight task is never
destroyed by a new comment. Manual dispatches share one group, so they
also serialize against each other and against issue work.

### Security boundaries

The agent can push code and trigger CI, so execution is gated:

- **Only the repository owner** can start a task. The actor is
  compared against `github.repository_owner` before anything runs.
- A new issue must carry the `[OpenCode]` prefix.
- A continuation comment must start with a supported command.
- Issue and comment text is untrusted. It is passed to the model as
  delimited task content, never as a shell command, and it cannot
  change the rules or the retry budget.
- The agent must never read, print or commit secrets, and must never
  write credentials into source.
- The agent must never implement LinkedIn scraping and must never use
  LinkedIn credentials.
- `.opencode/agents/enricher.md` is a deny-all enrichment tool. The
  agent is told not to reuse it, not to weaken it, and not to let
  another agent inherit it.
- Tests may not be deleted or weakened, and CI may not be bypassed.

Permissions granted to the agent job:

| Permission | Why |
| --- | --- |
| `contents: write` | push the agent's commit |
| `issues: write` | post the task report |
| `actions: write` | dispatch and read the validation run |

Pages deployment permissions (`pages: write`, `id-token: write`) stay
exclusively in the existing `Run Python Workers` workflow. The agent
cannot deploy Pages and cannot modify that workflow's permissions.

### OpenCode execution

- CLI: `@opencode/cli` pinned to **2.0.20**, installed on the runner
- Model: `opencode/space-bunny-free`
- Agent: `remote-engineer`, defined in
  `.opencode/agents/remote-engineer.md`
- Invocation: `opencode run --standalone --auto --format json
  --model opencode/space-bunny-free --agent remote-engineer "<prompt>"`

`--auto` approves permissions that are not explicitly denied. The
enricher agent denies everything, so it stays inert.

### What still requires a human

The agent reports BLOCKED and stops when it genuinely cannot
continue:

- a required secret is unavailable
- authentication needs a human, such as a login or an approval
- external infrastructure is down
- the task is destructive and ambiguous
- a permission problem it cannot repair

Everything else it attempts itself, up to the retry budget.

### Content ingestion boundary

LinkedIn content must remain manually or explicitly authorisedly
ingested. This control plane does not authorise scraping, and the
agent is explicitly instructed not to add it. Posts under `data/posts/`
are added by hand or by an authorised process.

### Running the control plane logic locally

```
python -m src.agent.preflight \
  --event-name issues \
  --actor <your-login> \
  --owner <repo-owner> \
  --title "[OpenCode] my task" \
  --issue-number 1 \
  --out trigger.json
```

Exit code `0` means authorized, `78` means the event must be ignored.
