"""
Revision units: one page per thing a candidate revises.

The taxonomy is 14 subjects and 154 subtopics because placement needs
154 buckets to put every question in exactly one. Revision does not work
that way. Nobody sits down to revise "SQL / Filtering and Set
Operations", then "SQL / PIVOT and UNPIVOT", then "SQL / Per-Group
Ranking and Top-N" -- those are three corners of one topic. Nor does
nobody revise "Databricks / Auto Loader and Ingestion" separately from
"Databricks / Delta Lake"; that is one lakehouse, and the two are only
separate because the corpus happened to talk about them separately.

So this module sits above the taxonomy and maps its subtopics onto
**revision units**: a coherent body of interview knowledge, one page.
The unit table below is the whole mapping, written out by hand, because
the judgement of "would a person revise these together" is not
mechanical and deriving it from label overlap would reproduce the
taxonomy's grain.

Nothing is lost by the merge. Every subtopic's questions, concepts and
sources land in its unit; a unit page carries the subtopics it covers as
sections, so a reader can still navigate to the finer distinction if they
want it.

**Pagination.** A unit stays on one page until it stops being
comfortable to revise, and then it splits into pages of the *same* unit
-- "SQL / Query Writing, page 1 of 4" -- rather than becoming four
topics. The threshold is a question count, which is measurable and
deterministic, so the same corpus always produces the same pagination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


#: A unit splits once it passes this many questions.
#:
#: Chosen from the measured distribution rather than taste: the median
#: unit holds 42 questions and the largest holds 487, so this keeps nine
#: units whole and splits the rest into pages of a readable length.
#: Raising it to 100 would fold SQL's five units' worth of material into
#: pages nothing can be scanned on; dropping it to 20 would turn the
#: small units into single-question pages, which is the per-subtopic
#: granularity this change exists to remove.
QUESTIONS_PER_PAGE = 40


@dataclass(frozen=True)
class UnitSpec:
    """One revision unit, as declared."""

    slug: str
    title: str
    #: The subject heading it is filed under on the home page.
    group: str
    #: One sentence a reader can revise from.
    summary: str
    #: The taxonomy subtopics it covers, as ``(subject_slug, subtopic_slug)``.
    sources: tuple[tuple[str, str], ...]

    @property
    def page_slug(self) -> str:
        return f"{self.slug}.html"


#: The units, in the order a candidate would work through them.
#:
#: Order is the order of the home page: SQL and Python first because they
#: come up in almost every data interview, then the platforms, then the
#: wider practice.
UNITS: tuple[UnitSpec, ...] = (
    UnitSpec(
        slug="sql-joins",
        title="SQL: Joins",
        group="SQL",
        summary=(
            "Inner, outer, left anti and left semi joins, self joins, and "
            "the cardinality mistakes that make a join quietly wrong."
        ),
        sources=(("sql", "joins"),),
    ),
    UnitSpec(
        slug="sql-window-functions",
        title="SQL: Window Functions",
        group="SQL",
        summary=(
            "ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG and LEAD; running "
            "totals; per-group top-N; and when a window beats a GROUP BY."
        ),
        sources=(
            ("sql", "window-functions"),
            ("sql", "per-group-ranking-and-top-n"),
            ("spark-and-pyspark", "window-functions-in-spark"),
        ),
    ),
    UnitSpec(
        slug="sql-aggregations",
        title="SQL: Aggregations, Filtering and Functions",
        group="SQL",
        summary=(
            "GROUP BY and HAVING, aggregate functions, per-group "
            "statistics, filtering and set operators, PIVOT and UNPIVOT, "
            "and the date, string and numeric functions that come up in "
            "interview questions."
        ),
        sources=(
            ("sql", "aggregation-and-group-by"),
            ("sql", "filtering-and-set-operations"),
            ("sql", "string-date-and-numeric-functions"),
            ("sql", "pivot-and-unpivot"),
        ),
    ),
    UnitSpec(
        slug="sql-ctes-and-subqueries",
        title="SQL: CTEs, Subqueries and Reusable Logic",
        group="SQL",
        summary=(
            "Common table expressions, correlated and uncorrelated "
            "subqueries, recursive CTEs, views, stored procedures and "
            "window functions as reusable definitions."
        ),
        sources=(
            ("sql", "ctes-and-subqueries"),
            ("sql", "views-and-stored-procedures"),
        ),
    ),
    UnitSpec(
        slug="sql-writing-and-optimization",
        title="SQL: Writing and Optimizing Queries",
        group="SQL",
        summary=(
            "Decomposing an under-specified question into a query, "
            "reading a query someone else wrote, tuning a slow one, and "
            "what actually makes a query fast."
        ),
        sources=(
            ("sql", "query-writing-and-problem-solving"),
            ("sql", "query-optimization"),
            ("sql", "query-reading-and-debugging"),
            ("sql", "transactions-and-locking"),
            ("sql", "sql-internals-and-execution"),
            ("sql", "sql-security"),
        ),
    ),
    UnitSpec(
        slug="sql-foundations",
        title="SQL: Foundations and Data Modelling",
        group="SQL",
        summary=(
            "What SQL is for, DDL and DML, data types and constraints, "
            "keys and normalisation, and the handful of questions asked "
            "before anything harder."
        ),
        sources=(
            ("sql", "sql-fundamentals"),
            ("sql", "data-types-and-constraints"),
            ("sql", "sql-against-external-data"),
            ("sql", "sql-and-nosql-choice"),
            ("sql", "general"),
        ),
    ),
    UnitSpec(
        slug="python-for-data-engineering",
        title="Python for Data Engineering",
        group="Python",
        summary=(
            "Lists, dictionaries, sets, tuples, functions, generators, "
            "comprehensions, OOP, modules and error handling; reading files "
            "larger than memory; and pandas for the dataframe work an "
            "interviewer asks about."
        ),
        sources=(
            ("python", "data-structures"),
            ("python", "functions-and-modules"),
            ("python", "pandas-and-dataframes"),
            ("python", "object-oriented-programming"),
            ("python", "language-fundamentals"),
            ("python", "files-and-large-data"),
            ("python", "files-and-io"),
            ("python", "typing-and-testing"),
            ("python", "error-handling"),
            ("python", "general"),
        ),
    ),
    UnitSpec(
        slug="spark-and-pyspark",
        title="Spark and PySpark",
        group="Spark / PySpark",
        summary=(
            "The DataFrame API, lazy evaluation, joins, partitioning and "
            "shuffle, caching and broadcast, Catalyst, how a job becomes "
            "stages, and the PySpark questions that come up in interviews."
        ),
        sources=(
            ("spark-and-pyspark", "dataframes-and-rdds"),
            ("spark-and-pyspark", "transformations-and-actions"),
            ("spark-and-pyspark", "joins"),
            ("spark-and-pyspark", "partitions-and-partitioning"),
            ("spark-and-pyspark", "shuffle-and-performance"),
            ("spark-and-pyspark", "caching-and-persistence"),
            ("spark-and-pyspark", "architecture-and-execution"),
            ("spark-and-pyspark", "catalyst-and-execution"),
            ("spark-and-pyspark", "spark-sql"),
            ("spark-and-pyspark", "deployment-and-operations"),
            ("spark-and-pyspark", "configuration-and-sizing"),
            ("spark-and-pyspark", "structured-streaming"),
            ("spark-and-pyspark", "spark-fundamentals"),
            ("spark-and-pyspark", "pyspark-coding-problems"),
            ("spark-and-pyspark", "ecosystem-integration"),
            ("spark-and-pyspark", "spark-versus-mapreduce"),
            ("spark-and-pyspark", "spark-projects-and-experience"),
            ("spark-and-pyspark", "general"),
        ),
    ),
    UnitSpec(
        slug="databricks",
        title="Databricks",
        group="Databricks",
        summary=(
            "Delta Lake, Unity Catalog, Auto Loader, Delta Live Tables, "
            "workflows and jobs, schema evolution, the lakehouse model, "
            "and when Databricks rather than Spark or a warehouse."
        ),
        sources=(
            ("databricks", "delta-lake"),
            ("databricks", "unity-catalog"),
            ("databricks", "workflows-and-jobs"),
            ("databricks", "schema-evolution"),
            ("databricks", "performance"),
            ("databricks", "auto-loader-and-ingestion"),
            ("databricks", "delta-live-tables-and-etl"),
            ("databricks", "lakehouse-architecture"),
            ("databricks", "file-formats"),
            ("databricks", "databricks-fundamentals"),
            ("databricks", "managed-tables-and-sql-analytics"),
            ("databricks", "integration-with-azure"),
            ("databricks", "platform-comparison-and-choice"),
            ("databricks", "databricks-projects-and-experience"),
            ("databricks", "general"),
        ),
    ),
    UnitSpec(
        slug="azure-data-factory",
        title="Azure Data Factory",
        group="Azure",
        summary=(
            "Pipelines and activities, data flows, linked services and "
            "datasets, parameters, triggers, incremental loads with a "
            "watermark, integration runtimes, monitoring and error "
            "handling."
        ),
        sources=(
            ("azure-data-factory", "pipelines"),
            ("azure-data-factory", "activities"),
            ("azure-data-factory", "data-flows-and-transformations"),
            ("azure-data-factory", "linked-services-and-datasets"),
            ("azure-data-factory", "parameters-and-variables"),
            ("azure-data-factory", "triggers"),
            ("azure-data-factory", "incremental-loads"),
            ("azure-data-factory", "integration-runtime"),
            ("azure-data-factory", "monitoring-and-error-handling"),
            ("azure-data-factory", "integration-with-databricks"),
            ("azure-data-factory", "general"),
        ),
    ),
    UnitSpec(
        slug="azure-platform",
        title="Azure Data Platform",
        group="Azure",
        summary=(
            "ADLS and blob tiers, Key Vault and managed identity, Event "
            "Hub and Event Grid, Synapse, Azure SQL Database, and how the "
            "services fit together."
        ),
        sources=(
            ("azure-platform", "storage"),
            ("azure-platform", "identity-and-secrets"),
            ("azure-platform", "stream-and-messaging"),
            ("azure-platform", "analytics-and-compute"),
            ("azure-platform", "integration-across-azure-services"),
            ("azure-platform", "multi-cloud-comparison"),
            ("azure-platform", "general"),
        ),
    ),
    UnitSpec(
        slug="aws-for-data-engineering",
        title="AWS for Data Engineering",
        group="AWS",
        summary=(
            "S3 and storage lifecycle, Redshift, Snowflake, BigQuery, "
            "Glue, RDS, networking and IAM, monitoring, and the "
            "serverless pieces an interviewer asks about."
        ),
        sources=(
            ("aws-and-cloud", "storage"),
            ("aws-and-cloud", "data-warehouses"),
            ("aws-and-cloud", "glue-and-etl"),
            ("aws-and-cloud", "rds-and-managed-databases"),
            ("aws-and-cloud", "bigquery"),
            ("aws-and-cloud", "nosql-and-messaging"),
            ("aws-and-cloud", "compute"),
            ("aws-and-cloud", "networking-and-iam"),
            ("aws-and-cloud", "cloud-storage-and-identity"),
            ("aws-and-cloud", "monitoring-and-operations"),
            ("aws-and-cloud", "multi-cloud-comparison"),
            ("aws-and-cloud", "general"),
        ),
    ),
    UnitSpec(
        slug="etl-and-integration",
        title="ETL, ELT and Data Integration",
        group="Data Engineering",
        summary=(
            "Extract, transform, load; ELT and reverse ETL; change data "
            "capture; batch and micro-batch; and building a pipeline "
            "someone else can run."
        ),
        sources=(
            ("data-engineering", "etl-and-elt"),
            ("data-engineering", "change-data-capture"),
            ("data-engineering", "file-formats-and-encoding"),
        ),
    ),
    UnitSpec(
        slug="data-modelling-and-warehousing",
        title="Data Modelling and Warehousing",
        group="Data Engineering",
        summary=(
            "Star and snowflake schemas, fact and dimension tables, "
            "normalisation, dimensional modelling, lakehouse versus "
            "warehouse, and partitioning."
        ),
        sources=(
            ("data-engineering", "data-modeling"),
            ("data-engineering", "data-warehousing"),
            ("business-intelligence-and-reporting", "data-modeling-in-bi"),
        ),
    ),
    UnitSpec(
        slug="scd-and-data-history",
        title="Slowly Changing Dimensions",
        group="Data Engineering",
        summary=(
            "Type 1, Type 2 and Type 3, effective dating, surrogate keys, "
            "and choosing an approach when a dimension keeps changing."
        ),
        sources=(
            ("data-engineering", "slowly-changing-dimensions"),
        ),
    ),
    UnitSpec(
        slug="data-architecture",
        title="Data Architecture and Pipelines",
        group="Data Engineering",
        summary=(
            "Batch, streaming, lambda and kappa; orchestration and "
            "scheduling; idempotency and failure handling; data quality "
            "and validation; governance, lineage and PII."
        ),
        sources=(
            ("data-engineering", "architecture-and-patterns"),
            ("data-engineering", "orchestration-and-scheduling"),
            ("data-engineering", "pipelines-and-troubleshooting"),
            ("data-engineering", "data-quality"),
            ("data-engineering", "governance-compliance-and-audit"),
        ),
    ),
    UnitSpec(
        slug="data-engineering-in-practice",
        title="Data Engineering in Practice",
        group="Data Engineering",
        summary=(
            "How to describe a pipeline you built: the scale you handled, "
            "the trade-offs you made, and how you debugged it."
        ),
        sources=(
            ("data-engineering", "project-and-experience"),
            ("data-engineering", "general"),
        ),
    ),
    UnitSpec(
        slug="coding-and-algorithms",
        title="Coding and Algorithms",
        group="Data Engineering",
        summary=(
            "The coding half of an interview: arrays and strings, "
            "hashing, two pointers, sliding window, trees and graphs, "
            "dynamic programming, and the data-shaped puzzles that are "
            "really SQL asked as puzzles."
        ),
        sources=(
            ("data-structures-and-algorithms", "puzzles-over-data"),
            ("data-structures-and-algorithms", "arrays-and-strings"),
            ("data-structures-and-algorithms", "two-pointers-and-sliding-window"),
            ("data-structures-and-algorithms", "dynamic-programming"),
            ("data-structures-and-algorithms", "searching-and-sorting"),
            ("data-structures-and-algorithms", "trees-and-graphs"),
            ("data-structures-and-algorithms", "streams-and-counting-structures"),
            ("data-structures-and-algorithms", "specialised-structures"),
            ("data-structures-and-algorithms", "practice-and-progression"),
            ("data-structures-and-algorithms", "greedy-and-algorithm-families"),
            ("data-structures-and-algorithms", "linked-lists"),
            ("data-structures-and-algorithms", "general"),
        ),
    ),
    UnitSpec(
        slug="git-cicd-and-iac",
        title="Git, CI/CD and Infrastructure",
        group="Data Engineering",
        summary=(
            "Branching and merge conflicts, a CI/CD pipeline for data "
            "code, infrastructure as code, containers, and deployment."
        ),
        sources=(
            ("git-and-devops", "version-control-with-git"),
            ("git-and-devops", "ci-cd-for-data"),
            ("git-and-devops", "infrastructure-as-code"),
            ("git-and-devops", "containers-and-orchestration"),
            ("git-and-devops", "observability"),
            ("git-and-devops", "general"),
        ),
    ),
    UnitSpec(
        slug="business-intelligence",
        title="Business Intelligence and Reporting",
        group="Data Engineering",
        summary=(
            "Dashboards and reports, DAX and measures, Power BI data "
            "modelling, semantic layers, and presenting a finding to "
            "someone who does not write queries."
        ),
        sources=(
            ("business-intelligence-and-reporting", "reporting-and-dashboards"),
            ("business-intelligence-and-reporting", "dax-and-measures"),
            ("business-intelligence-and-reporting", "spreadsheets-as-a-data-tool"),
            ("business-intelligence-and-reporting", "fabric-and-synapse"),
            ("business-intelligence-and-reporting", "communicating-with-data"),
            ("business-intelligence-and-reporting", "general"),
        ),
    ),
    UnitSpec(
        slug="machine-learning-and-ai",
        title="Machine Learning and AI",
        group="Data Science",
        summary=(
            "Supervised and unsupervised learning, feature engineering, "
            "model evaluation and deployment, deep learning and NLP, and "
            "generative AI -- at the depth a data interview asks about."
        ),
        sources=(
            ("machine-learning-and-ai", "supervised-learning"),
            ("machine-learning-and-ai", "unsupervised-learning"),
            ("machine-learning-and-ai", "feature-engineering-and-pipelines"),
            ("machine-learning-and-ai", "model-evaluation-and-deployment"),
            ("machine-learning-and-ai", "model-deployment-in-practice"),
            ("machine-learning-and-ai", "deep-learning-and-nlp"),
            ("machine-learning-and-ai", "generative-ai-and-agents"),
            ("machine-learning-and-ai", "fine-tuning-and-alignment"),
            ("machine-learning-and-ai", "classification-and-ranking"),
            ("machine-learning-and-ai", "ml-roadmaps-and-curriculum"),
            ("machine-learning-and-ai", "general"),
        ),
    ),
    UnitSpec(
        slug="system-design-and-scenarios",
        title="System Design and Scenarios",
        group="Interview Technique",
        summary=(
            "How to reason about a pipeline you have not built: "
            "trade-offs, failure modes, sizing, and the questions an "
            "interviewer asks to see how you think."
        ),
        sources=(
            ("interview-preparation", "system-design"),
            ("interview-preparation", "architecture"),
            ("interview-preparation", "scenario-questions"),
            ("interview-preparation", "troubleshooting"),
        ),
    ),
)


def page_count(question_count: int, per_page: int = QUESTIONS_PER_PAGE) -> int:
    """How many pages a unit of this size needs. Never fewer than one."""

    return max(1, math.ceil(question_count / per_page))


def group_order(groups: list[str]) -> list[str]:
    """Groups in the order their first unit appears."""

    seen: list[str] = []

    for unit in UNITS:
        if unit.group not in seen:
            seen.append(unit.group)

    return seen