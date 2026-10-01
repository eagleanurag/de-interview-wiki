# Delta Lake gives a data lake ACID guarantees

Delta Lake stores a transaction log beside the data, so a commit is atomic and a reader never sees a half-written table. Schema evolution lets you add, rename or drop a column without rewriting the files that are already there. Time travel queries an earlier version of the table by version number, which is how you compare a result against what the table looked like when the job ran.

On Databricks these tables sit on Apache Spark, and the file layout is what makes predicate pushdown and file skipping possible at all.
