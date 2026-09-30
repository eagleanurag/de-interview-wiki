# de-interview-wiki

Data Engineering interview knowledge base, built from normalized posts.

## Pipeline

The cloud pipeline is defined in
[`.github/workflows/run-python-worker.yml`](.github/workflows/run-python-worker.yml)
and runs in three stages:

1. **Discover** — scans `data/posts/*/post.json` and builds the
   matrix. No post list is hardcoded, so adding a post directory is
   enough to add a worker.
2. **Worker** — one matrix job per post. Each job creates a job
   manifest, runs the Python worker (`src/workers/cli.py`) to enrich the
   post through OpenCode + Space Bunny, and uploads its own artifact
   named `cloud-worker-result-<post_id>`. Workers never write to the
   repository and never push commits.
3. **Aggregate** — runs only after every worker succeeds. It downloads
   the worker artifacts, runs the Python aggregator, and uploads a
   single canonical `knowledge_base.json` as artifact
   `knowledge-base-<run_id>`.

## Ingestion

Manually captured posts are added with the ingestion layer in
[`src/ingestion/`](src/ingestion). A post is one directory:

```
data/posts/<post_id>/
├── post.json     authored content
└── media/        screenshots, PDFs
```

`post.json` uses the structure the repository already ships, so an
existing post can be edited by hand and a scaffolded one is
indistinguishable from a hand-written one.

### Adding a post

The quickest path is one command per capture. A *capture bundle* is
whatever a person has after reading a post by hand: some notes, a
screenshot, maybe a PDF.

```
python -m src.ingestion.cli import 2026-01-01-spark-shuffle \
    ~/captures/spark-shuffle \
    --platform manual --author "Interviewer" \
    --primary-topic "Apache Spark" --interview-relevant
```

A bundle may contain:

| File | Meaning |
| --- | --- |
| `notes.md`, `notes.txt`, `post.md`, … | becomes `original_text` |
| any other file | copied into `media/` and declared in `post.json` |
| `post.json` | used as the base document, questions included |

The same thing in two steps, when the capture arrives piecemeal:

```
python -m src.ingestion.cli new 2026-01-02-delta \
    --text-file notes.md --primary-topic "Delta Lake"

python -m src.ingestion.cli add-media 2026-01-02-delta \
    screenshot.png --description "The lineage diagram"
```

`list` shows what the pipeline will discover, and `validate` checks it:

```
python -m src.ingestion.cli list
python -m src.ingestion.cli validate
```

`validate` is the same check the test suite runs over the committed
posts, so running it is how you find out whether a post is ready to
commit. It exits non-zero when a post has an error, and only warns
about posts that will work but are described loosely.

### Behaviour that matters

- **Idempotent.** Re-running an import refreshes the authored fields
  and reports unchanged media instead of duplicating it, so a capture
  can be imported again after one more screenshot is added.
- **Non-destructive.** Enrichment output is never overwritten by an
  import. Adding a media file whose name exists with different content
  is refused unless `--force` is given.
- **Portable.** Declared media paths are relative to the post
  directory, and a path that would escape it is rejected, so a
  committed post can never point the loader at a file elsewhere.
- **Validated before it is written.** An import that would produce an
  unusable post is refused before anything is touched. `validate`
  checks the whole tree on demand, and the test suite validates the
  committed posts on every run, so a post that would break a worker
  cannot reach `main` unnoticed.
- **Local only.** The layer has no network client, no scraper and
  nowhere to put a credential. Content is added by hand or by an
  explicitly authorised process.

### Post ids

A post id becomes a directory name, part of a worker job id and a URL
segment, so it must be lowercase, start with a letter or digit, and use
only letters, digits, `.`, `-` and `_`. The importer refuses anything
else, and refuses an id that disagrees with the directory it lives in.

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

`tests/test_ingestion.py` covers the ingestion layer: post ids, the
authored document, media declaration and conflicts, idempotent
imports, validation, the CLI, and compatibility with the posts already
committed under `data/posts/`.

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
prompt construction, credential redaction, validation-run resolution,
task-outcome classification and report rendering.

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

Status: SUCCESS | SUCCESS_NO_CHANGES | DIRTY_NO_COMMIT
        | PUSH_FAILED | PUSH_UNVERIFIED | FAILED
        | BLOCKED | BLOCKED_AFTER_3_ATTEMPTS
Why / Task / Commit / Tests / Validation workflow / CI result
Recovery attempts: n/3
Files changed / Final result / Human action required
```

The same information is written to the GitHub Actions job summary.
Full OpenCode output, the exact prompt, the validation logs and the
report are kept as workflow artifacts:

- `opencode-agent-logs-<run_id>` — stdout, stderr and prompt, 14 days
- `opencode-report-<run_id>` — report and failure evidence, 30 days

### How a task's status is decided

A clean exit from the agent process is **not** a success on its own.
The status is derived from observed repository state, so the outcomes
that look alike from the outside stay apart:

| Status | Meaning |
| --- | --- |
| `SUCCESS` | implemented, committed and confirmed pushed |
| `SUCCESS_NO_CHANGES` | the agent exited cleanly and changed nothing, which is only valid for a read-only task |
| `DIRTY_NO_COMMIT` | the agent exited cleanly but left changes uncommitted, so the work died with the runner |
| `PUSH_FAILED` | a commit exists locally but is not on the remote branch |
| `PUSH_UNVERIFIED` | the commit exists but the push could not be confirmed |
| `FAILED` | the agent failed, or the test suite did not pass |

A commit counts as pushed only once the remote branch is confirmed to
contain it, so a local commit is never mistaken for a delivered one and
validation is never dispatched for a commit the remote has not seen. A
failing test suite also blocks `SUCCESS`.

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

### The workflow-file boundary

`GITHUB_TOKEN` **cannot** update a file under `.github/workflows/`. The
token is a GitHub App installation token, the app does not hold the
"Workflows" repository permission, and GitHub refuses the ref update:

```
! [remote rejected] main -> main (refusing to allow a GitHub App to
create or update workflow `.github/workflows/<file>.yml` without
`workflows` permission)
```

`workflows` is a GitHub App repository permission, not a
`GITHUB_TOKEN` scope. It is therefore not a valid key in a
`permissions:` block at all, and no workflow change can grant it. This
was verified against this repository: a source-only push succeeds with
the built-in token, a push that also touches a workflow file is refused,
and the same refusal happens when git authenticates through an askpass
helper instead of the checkout credential, so the limit is
authorization rather than authentication method.

**Manual configuration required (once, by the repository owner).** The
control plane reads an optional external credential from the
`AGENT_PUSH_TOKEN` environment variable, which the workflow fills from
the `OPENCODE_AGENT_TOKEN` repository secret:

1. Create a fine-grained personal access token: **Settings →
   Developer settings → Personal access tokens → Fine-grained
   tokens → Generate new token**.
2. Restrict it to **this repository only**, and set an expiry.
3. Grant these repository permissions: **Contents: Read and write**,
   **Workflows: Read and write**, **Pull requests: Read and write**,
   **Issues: Read and write**, **Actions: Read and write**, **Checks:
   Read and write**, **Commit statuses: Read and write**.
4. Store the value as the repository secret `OPENCODE_AGENT_TOKEN`
   (**Settings → Secrets and variables → Actions → New repository
   secret**).

The credential is optional. Without it the control plane behaves
exactly as before, and a refused workflow push is reported as BLOCKED
with this configuration request rather than worked around. With it, the
run log records `PUSH_AUTHENTICATION=external repository credential
armed` and workflow-file pushes succeed.

The value is never printed, logged, committed or passed on a command
line. `src/agent/credentials.py` writes a git askpass helper that
contains no secret and reads the environment variable at call time; the
helper lives in `.git/`, which is never committed. The environment
variable name contains `TOKEN`, so the existing redaction helpers
scrub the value from agent logs and artifacts as well.

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

The ingestion layer is local by design: it has no network client, no
scraper and nowhere to put a credential, so it can only import content
that is already on the machine. `platform` records provenance, so a
post added by hand says so rather than implying a named collection
process.

Its modules:

| Module | Responsibility |
| --- | --- |
| `src/ingestion/post_document.py` | the `post.json` shape, atomic writes, media declarations |
| `src/ingestion/importer.py` | discovery, creating, updating and importing posts |
| `src/ingestion/validation.py` | the post contract, and every check against it |
| `src/ingestion/post_loader.py` | reading a post for the worker |
| `src/ingestion/cli.py` | the command line above |

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
