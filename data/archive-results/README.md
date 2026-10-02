{
  "note": "Superseded enrichment, kept for the record and deliberately outside aggregation.",
  "why": [
    "This is an enrichment of sample_001 produced before the source-grounding",
    "check existed, under an older enricher contract. The current result for the",
    "same post is ../cloud_worker_sample_001.json, which carries the fingerprint",
    "that makes reuse decidable.",
    "It sat beside the live results under a name the aggregator's",
    "cloud_worker_*.json glob did not match, so it was invisible to the build and",
    "readable as if it were current. Moving it here keeps the data and removes the",
    "ambiguity; deleting it would have thrown away a real result for no gain.",
    "Re-running enrichment reproduces a current result for this post from the",
    "committed post, so nothing here is the only copy of anything."
  ]
}