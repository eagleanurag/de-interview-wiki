# Ingesting a change stream without double counting

The scenario: a Kafka topic carries change data capture events from a transactional database, and they have to land in a Delta Lake table exactly once. Writing to Delta Lake gives atomic commits, so a replayed batch converges on the same table rather than duplicating rows, which is the property that makes at-least-once delivery safe here.

The part that goes wrong in practice is offset management. A consumer that records its offset after writing, rather than in the same transaction as the write, re-reads a batch on a crash and the idempotent table absorbs it. One that records before writing loses it instead, and nothing notices until someone queries for a range that should have data.
