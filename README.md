# de-interview-wiki

Data Engineering interview knowledge base, built from normalized posts.

## Architecture

```
authorized source          LinkedInSource · ManualSource · SavedItemsSource
        ↓
collection                bounded, resumable, read-only
        ↓
data/posts/<id>/post.json source text, provenance, media
        ↓
enrichment                one worker per post, validated output,
                          checked against the post it came from
        ↓
consolidation             topics · concepts · technologies · questions
        ↓
canonical knowledge base   one authoritative JSON
        ↓
static wiki               posts · topics · concepts · technologies ·
                          questions · saved items · search
        ↓
GitHub Pages
```

A local saved-post archive enters at the second step, not the first.
`linkedin-archive` reads files already on the machine and writes them
out in the form `SavedItemsSource` reads, so there is one ingestion
path, one manifest and one set of post models rather than two of each.

Every source implements one contract in
[`src/ingestion/sources/base.py`](src/ingestion/sources/base.py), so the
pipeline never depends on a source being LinkedIn. Adding a source means
adding a class, not editing the collector.

Only `LinkedInSource` touches a website, and only under an
already-authorized sign-in. `ManualSource` and `SavedItemsSource` read
files you put in a directory, which is what keeps the rest of the
pipeline testable and keeps it working when a web interface changes.

## Running the whole pipeline locally

```bash
python -m src.pipeline                    # discover → enrich → aggregate → site
python -m src.pipeline --only enrich      # one stage
python -m src.pipeline --force-enrich     # re-enrich everything
python -m src.pipeline --jobs 6           # six posts at a time
```

`--jobs` runs several posts at once. Each post is already its own
worker result and CI already runs one job per post, so the work is
independent; this applies the same fan-out locally. One is the default,
because a bound nobody asked for is a surprise. With several hundred
posts the difference is the difference between hours and minutes.

Output goes under `build/`, which is git-ignored, so nothing in this
command writes into the committed repository. Enrichment is the only
stage that touches untrusted model output, and one post failing costs
that post rather than the run.

Enrichment is incremental. Each result records a fingerprint of the
content it describes, so a post whose text and media have not changed
keeps the analysis it has and the model is not called again. Editing a
post, or changing the enrichment contract, invalidates exactly the
results that no longer apply.

### How the cloud pipeline batches

GitHub allows a matrix of at most **256 jobs per run**, so a local
archive — which took this repository from seven posts to nearly five
hundred — could not be enriched one job per post. The workflow
therefore groups posts into batches of 25 and gives each job a whole
batch. The job count is a function of the batch size rather than of
the post count, which is what lets the workflow keep working as
material is added.

One post failing still costs one post: every post in a batch is
attempted, the failures are named, and the results that did land are
uploaded regardless, so aggregation reports the shortfall instead of
nothing at all. Aggregation compares the count it received against the
count discovered and fails loudly when they differ.

Set `WORKER_BATCH_SIZE` as a repository variable to change the batch
size. The default is 25. If a repository ever grows past the limit
again, the workflow refuses to start and says what to raise it to,
rather than being cancelled without explanation.

## Pipeline

The cloud pipeline is defined in
[`.github/workflows/run-python-worker.yml`](.github/workflows/run-python-worker.yml)
and runs in four stages:

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
4. **Generate and deploy** — renders the static site and publishes it to
   GitHub Pages.

## Adding your own material

The pipeline is not LinkedIn-specific. Anything you have already
captured and are authorized to use can go in.

Put a bundle in `data/incoming/`. A bundle is a directory holding your
text and any media beside it:

```
data/incoming/
    spark-partitioning/
        notes.md
        query-plan.png
    slow-query-investigation/
        notes.md
        traces.pdf
    capture.json
```

Then:

```bash
python -m src.ingestion.collect_cli run --source manual --dry-run
python -m src.ingestion.collect_cli run --source manual
python -m src.pipeline
```

Nothing needs to be written as JSON. A directory of notes becomes one
post, an image or PDF beside it is attached, and a `.jsonl` export
becomes one post per line. Running it again refreshes what changed
rather than creating a second copy.

| In a bundle | Read as |
|---|---|
| `capture.json` or `post.json` | the post, with the metadata it declares |
| `*.md`, `*.txt` | the post's text |
| `*.png`, `*.jpg`, `*.webp`, `*.gif` | attached media |
| `*.pdf` | attached media; its text is extracted |
| `*.jsonl` at the root | one post per line |

A directory with only a PDF or only an image is still a post: it
carries its media, and the text appears once the media stage reads it.

`data/incoming/` is git-ignored. Captured material may be large, private
or already published somewhere, and none of that belongs in this
repository's history. Only the ingested result under `data/posts/` is
committed.

### Identity and duplicates

A bundle that declares its own identifier keeps it. One that does not is
identified by a digest of its content and its media, with a readable
prefix taken from its opening words. Two copies of the same capture
therefore resolve to the same post however they are named, and editing a
file makes it a different post rather than a silently ignored duplicate.

Nothing is invented. A PDF that cannot be read records why and carries
no text, and a post with no analysis is still stored, labelled and
reachable rather than dropped.

## Saved Items

A saved list is not a document. It is a list of links, and most of them
have no body text behind them — LinkedIn gives you a URL and a date and
nothing else. This is the workflow for that case, and the rule it keeps
is the one that matters: **a link with nothing behind it stays a link.**

It needs no credentials, no browser and no network. Nothing is fetched
to fill a gap, so the pipeline does not depend on how LinkedIn's web
interface is laid out today, and a change to it cannot break this.

Everything here is a local folder you control. The project reads what
you put in it; it never goes and gets anything.

### The whole workflow

```bash
# 1. Build the list from whatever you have.
python -m src.ingestion.collect_cli saved-items-init \
    --from my-saved-posts.txt

# 2. Put content you captured yourself beside it, one folder per post.
#    data/incoming/saved-items/captures/<source_id>/content.md

# 3. Check the inbox and read every problem it reports.
python -m src.ingestion.collect_cli saved-items-validate

# 4. See exactly what would happen. Writes nothing at all.
python -m src.ingestion.collect_cli saved-items --plan

# 5. Import.
python -m src.ingestion.collect_cli saved-items \
    --input data/incoming/saved-items/manifest.csv

# 6. Enrich, and generate the site.
python -m src.pipeline
```

`saved-items-status` tells you where you are at any point, and
`saved-items-validate` is the one to run first: it is the command that
tells you what is wrong rather than what happened.

### 1. Build the list

The command takes whatever you happen to have and turns it into the one
list the importer reads.

```bash
# A file of links, one per line.
python -m src.ingestion.collect_cli saved-items-init \
    --from saved-links.txt

# A spreadsheet. Columns are matched by name, so the header need not be
# exact.
python -m src.ingestion.collect_cli saved-items-init \
    --from saved-posts.csv

# A browser bookmark export.
python -m src.ingestion.collect_cli saved-items-init \
    --from bookmarks.html

# One link at a time.
python -m src.ingestion.collect_cli saved-items-init \
    --url https://www.linkedin.com/posts/...
```

It writes `data/incoming/saved-items/manifest.csv`. Running it again
with another source **merges** rather than replaces, so two exports
produce the union and not whichever was read last. It never touches a
capture and never touches an enrichment, under any flag.

A PDF of saved posts is refused, with the reason: which line of a
paginated document belongs to which post cannot be established
reliably, and attaching the wrong text to the right link is worse than
not importing.

Column names, all matched case-insensitively:

| Meaning | Accepted as |
|---|---|
| the link | `url`, `URL`, `link`, `LinkedIn URL`, `Saved URL`, `permalink` |
| when you saved it | `saved date`, `date saved`, `saved on`, `date`, `created at` |
| title | `title`, `headline`, `subject` |
| author | `author`, `saved by`, `owner` |
| your note | `notes`, `note`, `comment` |
| a capture folder | `bundle`, `capture`, `content`, `folder` |

A column named for a credential — `password`, `token`, `cookie`,
`api_key`, `storage_state` — is refused and reported, never read. A
saved list does not contain credentials, and a file that does is not
something to copy into a knowledge base.

The same link written two ways is one item. Whitespace, a trailing
slash, a `#fragment` and tracking parameters are all normalised away;
the part that identifies the post is preserved exactly, so two different
posts never collapse into one.

### 2. Add the content you captured

Where you have the post itself, put it in a folder under `captures/`.
The folder claims one item, in whichever way is easiest:

```
data/incoming/saved-items/
    manifest.csv
    captures/
        urn-li-saved-3a22fed0239f2912/     # named after the item's id
            content.md
            screenshot.png
            document.pdf
        any-folder-name/
            capture.json                   # {"source_id": "urn:li:saved:..."}
            page.html
        another-folder/
            capture.json                   # {"url": "https://www.linkedin.com/..."}
            page.html
```

`captures/` is a container, not a capture. A folder inside it is one
saved post; nothing inside it needs to be listed anywhere, and a
screenshot beside a transcript needs no line in a manifest.

`saved-items-status --show-pending` prints the folder name for every item
still waiting for content, so you never have to work out an id yourself.

### 3. `capture.json`

One small file, and only the URL is required:

```json
{
  "url": "https://www.linkedin.com/posts/..."
}
```

Everything else is optional — `title`, `author`, `notes`, `captured_at`,
`text` — and a capture with one field is complete. Fields naming a
credential are dropped and reported rather than stored; the report names
the field so you can find it, and never quotes its value.

### 4. Check it first

```bash
python -m src.ingestion.collect_cli saved-items-validate
```

It checks the manifest, every URL, duplicate links, capture association,
path containment, corrupt files, credential-shaped fields, orphan
captures and missing captures. It exits non-zero when there is a real
error, and zero when there are only warnings — most of what you saved
has no capture yet, and that is a backlog rather than a fault. Add
`--strict` to treat warnings as errors.

Every problem is reported the same way, because a bare `ERROR` tells you
nothing you can act on:

```
ORPHAN_CAPTURE

Where:
  captures/example

Reason:
  No source_id or LinkedIn URL was found in this folder, so it cannot be
  attached to a saved item. Nothing was imported from it and nothing was
  deleted.

Fix:
  Add capture.json to this folder containing at least:

  {
    "url": "https://www.linkedin.com/posts/..."
  }

  Alternatively, rename the folder to the saved item's id.
```

Stack traces appear only with `--debug`.

### 5. Preview, then import

```bash
python -m src.ingestion.collect_cli saved-items --plan
```

`--plan` groups every item by what would happen to it — `NEW`,
`CHANGED`, `UNCHANGED`, `DUPLICATE`, `MISSING_CAPTURE`, `INVALID`,
`FAILED` — and lists the problems and orphan captures. `--dry-run` gives
the same answer as a summary. Neither writes anything: not a post, not
the manifest, not a report file.

Then import for real. `--adopt-orphans` creates a saved item for a
capture that names a usable LinkedIn URL but has no manifest item yet. A
capture with no usable URL is still reported, never adopted: attaching it
would mean inventing the post it belongs to.

Useful options: `--report FILE`, `--max-posts N`, `--json`,
`--posts-root DIR`.

### 6. Status

```bash
python -m src.ingestion.collect_cli saved-items-status
```

Every number is counted from the manifest, the captures on disk and the
posts that exist. Nothing is estimated, and a section with nothing in it
says so rather than being left out, because an absent number reads as a
forgotten one.

```
Saved Items
-----------
Manifest items              500
With captures                42
Metadata only               458
Ready to import              10
Already imported             32
Changed                       2
Failed                        0
Pending                     458

Capture quality
---------------
text                        25
text_and_media               8
image_only                   6
document_only                2
partial                      1

Content types
-------------
Markdown         25
HTML              5
PDF               8
Images           17

Knowledge
---------
Imported posts               40
Interview relevant           31
Questions generated        186
Topics                      27
Technologies                19
```

Add `--show-pending` to list the items still waiting for a capture, and
`--json` for the whole thing.

### What it will not do

- **It will not fetch anything.** A missing post stays missing. No
  LinkedIn page is ever requested to complete an item, and a resource a
  captured page names is never requested either.
- **It will not invent a body.** A URL and a date produce a saved item
  with a URL and a date. Only content you supplied becomes post text.
- **It will not invent a link.** A capture folder that says nothing about
  which post it is is reported, not guessed at.
- **It will not read a credential.** There is nothing here to
  authenticate with, so the code has no way to try.
- **It will not OCR.** An image is kept as an image and says so. Until
  you write the text beside it, it contributes no words.
- **It will not invent an interview question.** A post that supports no
  useful question produces none, and a question naming a technology the
  post never mentions is removed before it is stored. A hiring notice
  stays a hiring notice.
- **It will not act on LinkedIn.** No like, comment, share, follow,
  connection, message, post, delete or setting change. There is no
  Saved Posts crawler here and none is planned.

### Staying honest about provenance

Every imported post records `capture_method` — `user_provided`,
`user_saved_page` or `user_export` — and a `saved_item` block naming the
item, its save date, how the capture was matched, and what the capture
actually was. A post that came from an authorized collection run is not
the same thing as a post you supplied, and the difference survives into
the published wiki, where the post page says which it is and links back
to the item.

The capture quality is named rather than scored, because every value is
something the pipeline observed and none of them is a number it made up:

| Quality | What it means |
|---|---|
| `text` | The post's text was captured |
| `text_and_media` | Text, with a file or two beside it |
| `partial` | Text, but at least one supplied file could not be read |
| `image_only` | A screenshot, kept as an image; no text was read from it |
| `document_only` | A document was captured but yielded no text |
| `metadata_only` | A saved link with no content behind it |

Running it again imports nothing new: an item is identified by its
canonical URL, and only a capture whose content actually changed is
re-read. Editing one file re-enriches that one post.

On the published site, the **Saved Items** page lists what was actually
captured, grouped by how complete the capture was, and links each entry
to its post and back to the original source. It is one page rather than
one per item, because a saved list runs to hundreds of links and most of
them never get a capture — a link with nothing behind it has no page
worth reading, and it stays in your drop zone where you can act on it.

## A Local LinkedIn Archive

If you already have a saved-posts archive on this machine — a folder
holding `posts_archive.json` and a `media/` directory — this project can
read it. It writes the content out in the form the [Saved
Items](#saved-items) importer already reads, and everything downstream
is the same pipeline: same models, same enrichment, same aggregation,
same wiki, same search.

```bash
python -m src.ingestion.collect_cli linkedin-archive \
    --input "C:\path\to\linkedin_saved_archive" \
    --import
```

Then enrich and build, as usual:

```bash
python -m src.pipeline --jobs 6
```

### What this is, and what it is not

It **reads local files**: a JSON list and a folder of already-downloaded
images. It opens no browser, drives no automation, reads no credential,
reads no cookie jar, and makes no network request of any kind. The URLs
inside the archive are kept as provenance and are never treated as
instructions to fetch something.

The archive's own `chrome_session/` directory is named and skipped. A
browser session is a credential wearing a different shape, and a reader
should not have to know that to avoid it.

The archive is treated as read-only input. Nothing in this project opens
it for writing, and media is hard-linked into the working drop zone
rather than copied, so the archive stays the single copy of the original
bytes.

### The format it reads

```
linkedin_saved_archive/
├── posts_archive.json
├── media/
│   └── ...
└── chrome_session/          # present or not; never opened
```

A list of records. Only `post_id` is required; everything else is read
if it is there and ignored if it is not:

| Field | What it is used for |
|---|---|
| `post_id` | the archive's own identifier, kept as provenance |
| `permalink` | the LinkedIn link, when the archive has one |
| `text` | the post body, stored exactly as recorded |
| `author.name` / `.profile_url` / `.headline` | attribution |
| `scraped_at` | when the archive captured it |
| `relative_time` | the original relative form, kept verbatim |
| `media.saved_files` | names, resolved against `media/` |
| `media.original_urls` | provenance only; never fetched |

A record with no `permalink` is still imported. More than a third of a
real archive has none, and there is nothing in the media filenames or
media URLs that would recover it. Those posts are identified by the
archive's own `post_id` under a separate `urn:li:archive:` namespace,
and their pages render **no source link** rather than a link that goes
nowhere.

### What the report says

Every number below is counted off the archive as it was found. Nothing
is expected, nothing is assumed, and none of these figures is what the
command will print for a different archive. This is one real run:

```
Read 485 record(s) from C:\path\to\linkedin_saved_archive
  valid 485   unreadable 0   repeated content 2
  media 312 file(s), 23,125,048 bytes, 0 unreadable
  chrome_session/ is present and was not opened.

LinkedIn Archive
----------------
Records
-------
Total records                    485
Valid records                    485
Invalid records                      0
Duplicate records                   2
Unique records                   483

With a permalink                 315
With no permalink                170
With an author name              168
With no text                       0

Media
-----
Records with media               312
Records without media            173
Records missing a file             0

Files in the archive             312
Total bytes                  23,125,048
Images                           312
Named for a different type        41
Identical to another file         21
```

Every number is counted off the archive as read. A section with nothing
in it says zero rather than being left out.

`Named for a different type` is worth reading rather than skipping: a
real archive named every asset `.jpg` and a third of them are PNG or
GIF. The importer detects the format from the bytes, keeps the name the
record refers to, and says so — a reader told a file is a JPEG and
handed a PNG has been told something false.

### One bad record costs one record

Every record is read inside its own isolation. A malformed row, an
unreadable file, a missing permalink or a media name that points
nowhere is recorded against that record and the rest continue. A
`post_id` that is missing is refused rather than given a synthetic one,
because a record the pipeline cannot name is one it will import again
on the next run.

### Duplicates

Two captures of one post would become two posts, so whole-text
repetition is collapsed. The digest covers the *whole* text: a prefix
fingerprint was tried first and folded together two posts that agree
for a few hundred characters and diverge afterwards, which is worse
than reporting a duplicate that is not one.

The surviving record is chosen by what is least costly to lose — a
permalink beats none, media beats none, then the longer text, then the
identifier. The record that lost is **not deleted**: it stays in the
archive and in the report, and the surviving record names it.

Media that is byte-identical across posts is detected and reported
rather than deduplicated away, because each post's media travels with
that post and a shared file would mean one disappearing when another is
removed.

### Re-running it

Re-running is cheap and safe. A prepared capture is left alone when its
text and its media digests are unchanged, and the import reports
`Already prepared` for each one. Change a post's text, or replace a
file under the same name, and that one post is re-prepared and
re-imported. A metadata change alone — a new capture timestamp, say —
re-prepares nothing, because there is nothing to re-read.

The whole thing is resumable: an interrupted run leaves what it
finished, and the next run does the rest.

### What it will not do

- **It will not fetch anything.** The URLs in the archive are
  provenance, never a download list.
- **It will not open a browser** or use the archive's session.
- **It will not read a credential**, a cookie, or the environment.
- **It will not invent a URL** for a post the archive has none for.
- **It will not rewrite a post's text.** It is stored as recorded.
- **It will not let a record reach outside the media folder.** A
  `..` segment, an absolute path, or a symbolic link is refused and
  reported, using only the filename.
- **It will not run anything it reads.** An SVG carrying a script is
  kept as a file and never executed.

### The command

```bash
python -m src.ingestion.collect_cli linkedin-archive \
    --input "C:\path\to\linkedin_saved_archive"     # required
    [--out data/incoming/linkedin-archive]            # the drop zone
    [--import]                                       # import as well
    [--include-duplicates]                           # keep repeats
    [--report FILE]                                  # JSON report
    [--posts-root DIR]                               # where posts land
    [--json] [--debug]
```

`--input` has no default. A path this project guessed at is a path this
project should not be reading.

Without `--import` the drop zone is written and nothing else happens,
so you can read the report first and decide:

```bash
python -m src.ingestion.collect_cli linkedin-archive --input "..." \
    --report archive-report.json

python -m src.ingestion.collect_cli saved-items-validate \
    --bundle-root data/incoming/linkedin-archive

python -m src.ingestion.collect_cli saved-items \
    --bundle-root data/incoming/linkedin-archive
```

Once the drop zone exists it is an ordinary saved-items inbox, so
`--plan`, `--dry-run` and `saved-items-status` all work on it.

An inbox has one contract: every supported file in it is a candidate
list. The importer writes a manifest and capture folders and nothing
else, because a side-car JSON beside the manifest is read as a list of
rows with no URL and every saved-items command then reports it. The
provenance it would have carried is already in the files the reader
understands — every row has an archive-namespaced `Source ID`, and every
`capture.json` names the archive post it came from — and the counts are
in the import report.

```bash
python -m src.ingestion.collect_cli saved-items-validate \
    --bundle-root data/incoming/linkedin-archive
```

### Its modules

| Module | Responsibility |
|---|---|
| `identity.py` | source ids, including for records with no permalink |
| `archive.py` | reading, validation, the media index, per-record isolation |
| `prepare.py` | writing the drop zone the existing importer reads |
| `report.py` | the numbers, as text or JSON |

The archive is not committed, and neither is its media. The drop zone
lives under `data/incoming/`, which is git-ignored, and the posts it
produces are committed like any other post.

## Collection

Collection reads an authorized source and writes normalized posts. It is
strictly read-only: it navigates, expands truncated text and submits the
sign-in form, and does nothing else.

```bash
# One-off sign-in. Fills the credentials in .env, and hands the browser
# to you only if LinkedIn presents a challenge.
python -m src.ingestion.collect_cli run --source linkedin --login --headed

# Collect, bounded and resumable.
python -m src.ingestion.collect_cli run --source linkedin --max-posts 10

python -m src.ingestion.collect_cli status
```

Configuration lives in a git-ignored `.env`:

```
LINKEDIN_USERNAME=
LINKEDIN_PASSWORD=
```

Credentials are read through
[`src/ingestion/credentials.py`](src/ingestion/credentials.py), which
reports only whether they are configured. Values are never printed,
logged, written to a checkpoint, or committed.

### Authentication

Sign-in is automatic: the configured username and password are filled and
submitted, then authentication is *verified* rather than assumed. A run
that cannot prove it is signed in does not save a session and does not
report success.

Every browser operation is bounded. If LinkedIn presents a CAPTCHA, OTP,
2FA or any other verification step, the run stops and asks the human to
complete it in the open browser; nothing here attempts to solve or work
around a challenge. After the human confirms, verification runs once,
also under a bound.

The saved session lives in the git-ignored `.agent/secrets/` tree. It
expires; when it does, the next run signs in again automatically and
saves a fresh one.

### Content kinds

A profile exposes more than posts. The collector walks each kind under one
shared budget:

- posts, keyed by their feed permalink URN
- authored articles, keyed by their slug

Both produce `data/posts/<id>/post.json` with full provenance, so a
refreshed post is updated in place and never duplicated.

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
  "schema_version": 2,
  "generated_at": "...",
  "stats": {
    "result_files_found": 7,
    "posts_aggregated": 7,
    "files_skipped": 0,
    "posts_enriched": 7,
    "posts_interview_relevant": 1,
    "topics_consolidated": 29,
    "concepts_consolidated": 19,
    "technologies_consolidated": 8,
    "questions_consolidated": 2
  },
  "skipped_files": [],
  "posts": [],
  "knowledge": {
    "topics": [],
    "concepts": [],
    "technologies": [],
    "questions": [],
    "content_kinds": {}
  }
}
```

### Consolidation

[`src/aggregation/consolidation.py`](src/aggregation/consolidation.py)
turns a list of posts into navigable knowledge. Topics, concepts and
technologies that several posts contribute to become one node that keeps
every post behind it, so a reader can go from a concept back to the
original text.

Nothing is invented. A topic only exists because a post recorded it, a
technology only appears when a post's text mentions it, and a question
only exists when enrichment produced one for material it actually read.
Consolidation refuses to publish a node referencing a post it never saw.

Labels that differ only in case or punctuation consolidate into one node,
and every node keeps its provenance, so merging never loses a source.

`content_kinds` records what each post *is* — `technical`,
`job_announcement`, `event`, `unenriched` and so on — so a post that
contributed no knowledge is labelled rather than silently dropped.

## Recovery

Every stage is resumable, and none of them overwrites work that is
already done.

| Interrupted at | What the next run does |
|---|---|
| Collection | Known posts are refreshed in place, not re-imported |
| Media | A file that failed is recorded and retried; the post survives |
| Enrichment | Posts whose fingerprint matches are reused |
| Aggregation | The knowledge base is written atomically via a `.tmp` file |
| Wiki | Output is staged and swapped in, so a partial site is never served |

A failure is reported, never swallowed. One unreadable PDF costs that
PDF. One rejected model response costs that post, and its source text
stays where it was.

`git status` is the first thing to check when something looks wrong. A
ref file that loses its contents leaves every file looking untracked and
the branch unreferenced, which is silent until something is committed on
top of it.

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

`tests/test_authentication.py` covers the authentication state machine:
credential handling, redaction, challenge routing, bounded verification,
and that a `KeyboardInterrupt` never yields a success exit code.

`tests/test_collection.py` covers the collector and the LinkedIn source:
bounded collection, deduplication, refresh-in-place, resume, and the
read-only guarantee.

`tests/test_page_scripts.py` runs the JavaScript the collector executes
against a **real DOM** through Playwright, so post extraction, submit
resolution, profile resolution and scroll-container detection are
verified as behaviour rather than as source strings. It skips
automatically when Chromium is unavailable.

`tests/test_consolidation.py` covers consolidation: grouping, case- and
punctuation-insensitive merging, provenance, technology recognition and
determinism.

`tests/test_pipeline_e2e.py` runs the whole pipeline — validate,
aggregate, generate — over the committed data and checks the properties a
reader depends on: every post reachable, every link relative, every link
resolving, and a byte-identical rebuild.

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
| `pull-requests: write` | the pull-request work the task contract offers |
| `checks: write` | publish the check run a task may report against |
| `statuses: write` | publish a commit status |

The report job is narrower still: `contents: read`, `issues: write`
and `actions: read`, the last one so it can actually fetch the failing
validation logs it reports. The preflight job holds `contents: read`
and `issues: read` only and can never write. There is no `write-all`
anywhere, and each grant is deliberate, because a job-level
`permissions` block replaces the top-level one rather than merging
with it.

Pages deployment permissions (`pages: write`, `id-token: write`) stay
exclusively in the existing `Run Python Workers` workflow. The agent
cannot deploy Pages and cannot modify that workflow's permissions.

The grant is also read back out of the workflow when the prompt is
built, so the prompt tells the agent what its token really holds
instead of a list that can drift away from the control plane.

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

**Manual configuration, done once by the repository owner.** The
control plane reads an optional external credential from the
`AGENT_PUSH_TOKEN` environment variable, which the workflow fills from
the `OPENCODE_AGENT_TOKEN` repository secret. It is currently
configured; recreate it if it is ever removed, using:

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

Only the step that hands the credential to git receives the value. The
step that builds the prompt receives
`${{ secrets.OPENCODE_AGENT_TOKEN != '' }}` instead, a boolean, so the
prompt can state the real workflow-file boundary without any step that
renders text ever holding the secret.

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

The saved-items importer holds to the same boundary. It reads a list you
exported and the captures you put beside it, and it cannot reach a
credential, open a connection or drive a browser. That is enforced by
tests that parse its imports rather than matching its text, so refusing
a column named `token` cannot be mistaken for using one. Nothing is
fetched to complete an item that arrived as a bare link, and no captcha,
MFA, OTP, rate limit or access control is bypassed, evaded or worked
around anywhere in this repository.

Its modules:

| Module | Responsibility |
| --- | --- |
| `src/ingestion/post_document.py` | the `post.json` shape, atomic writes, media declarations |
| `src/ingestion/importer.py` | discovery, creating, updating and importing posts |
| `src/ingestion/validation.py` | the post contract, and every check against it |
| `src/ingestion/post_loader.py` | reading a post for the worker |
| `src/ingestion/cli.py` | the command line above |
| `src/ingestion/saved_items/` | saved-list import: URL identity, capture contract, bundle matching, the manifest |
| `src/ingestion/linkedin_archive/` | reading a local saved-post archive, and writing it out as a saved-items drop zone |
| `src/ingestion/collect_cli.py` | `saved-items`, `saved-items-init`, `saved-items-status`, `saved-items-validate` and `linkedin-archive` |
| `src/ai/grounding.py` | checking generated questions against the post they came from |
| `src/wiki/saved_items.py` | the Saved Items page |

The saved-items package in full:

| Module | Responsibility |
| --- | --- |
| `urls.py` | normalisation, validation, stable source ids |
| `model.py` | one saved item, its state, and how two records merge |
| `readers.py` | CSV, TSV, TXT, JSON and JSONL saved lists |
| `capture.py` | the `capture.json` contract, and the fields it refuses |
| `bundles.py` | capture discovery, containment, extraction, fingerprints |
| `manifest.py` | the atomic record of what has been imported |
| `plan.py` | one read of the inbox, answering what a run would do |
| `diagnostics.py` | where, why, and the fix — for every problem |
| `intake.py` | building the list from links, a spreadsheet or bookmarks |
| `source.py` | the source contract the collector drives |

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
