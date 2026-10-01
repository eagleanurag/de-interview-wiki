# Delta Lake compaction

Compaction rewrites small files into fewer, larger files.

## Why it matters

Small files defeat parallelism: every task pays planning overhead before it reads anything, so throughput falls even when the data volume is unchanged.

## When to compact

- Many small files accumulate after a streaming write.
- A query is scanning file counts rather than bytes.

## Trade-off

Compaction rewrites data, so it costs compute and I/O. Doing it continuously can cost more than the reads it saves.
