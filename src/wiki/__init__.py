"""
Static Data Engineering interview wiki generator.

    python -m src.wiki.generator \
      --input aggregation/knowledge_base.json \
      --output site

Reads the canonical knowledge base produced by
`src.aggregation.aggregator` and writes a dependency-free static
site: a dashboard, client-side search, topic pages, a question browser
and one page per post.

Every link is relative to the page that contains it, so the same output
works from the filesystem, from a root-hosted site, and from a GitHub
Pages project path, with no base URL configuration.
"""
