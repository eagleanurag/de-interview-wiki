# Enrichment from GitHub Actions run 37026765769

482 post enrichments produced by the cloud worker fan-out, kept for the
record and deliberately outside aggregation.

## What they are

On 2 October 2026 this repository had 490 posts and no committed
enrichment for most of them. Enrichment ran as a GitHub Actions matrix:
twenty batches of twenty-five posts, three batches at a time, each
batch calling the model once per post through `src/workers/cli.py`.

Fourteen batches succeeded and six failed. Because a batch had failed,
the aggregate, wiki and deploy jobs were skipped and nothing was
published. All twenty artifacts survived, because the upload step runs
whether or not the batch did, which is the only reason there are 482
results rather than the 350 the successful batches alone would have
produced.

Every file here validates against the post models. None is corrupt.

## Why they are not in the knowledge base

Two reasons, both about honesty rather than tidiness.

**They add no coverage.** The local orchestrator had already enriched
all 490 posts before these were collected, and those results carry a
fingerprint of the content they describe. These do not: a file here is
a bare `KnowledgePost` with no `_enrichment` block, so there is no way
to ask whether it still describes its post. The local results can be
judged; these cannot.

**They are a second opinion, not a better one.** Each is an independent
draw from the same model against the same prompt. Comparing the two for
the same post shows the same substance in different words — and the
CI files carry no record of what source grounding removed, because that
report was written by the pipeline that produced them and these did not
go through it. Overwriting a local result with one of these would
replace a traceable record with an untraceable one.

## Why keep them at all

Because they are real model output for real posts, produced under
conditions nobody will be able to reproduce exactly, and because
discarding successful work on the grounds that it is redundant is a
decision that should be visible rather than silent. They also make one
comparison possible later that is not otherwise available: what the same
prompt produced across two runs, which is worth knowing before anyone
trusts a single draw.

## What is not here

The 490 job manifests under the same artifact names. They record what
was asked for rather than what came back, and the eight posts they name
that produced no result are recorded in the run's failure output rather
than here.

## Reproducing the current results

Nothing in this directory is the only copy of anything:

```bash
python -m src.pipeline --jobs 3
```

regenerates a current, fingerprinted result for every post from the
committed posts.