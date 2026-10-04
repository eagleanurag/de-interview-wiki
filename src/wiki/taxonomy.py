"""
The revision curriculum: a small, fixed hierarchy over a large label space.

The corpus produced 1,560 distinct topics, 3,294 subtopics and 3,106
concepts. That is the honest output of asking a model to label 490 posts,
and it is exactly why the site felt like an archive: two thousand and
forty-six top-level navigation entries is not navigable, and "Databricks
troubleshooting" sitting beside "Window functions" as peers tells a
revising reader nothing about where to look.

So the labels are kept -- none are discarded, they are the evidence --
and *placed*. This module is the placement table: eight major subjects,
each with an ordered set of subtopics, each with the phrases that select
it.

**Matching is on word boundaries, never substrings.** That is not a detail.
An early keyword sweep for ``orm`` matched 116 topic labels, every one of
them a false hit inside *perf**orm**ance*, *plat**orm*** and *transf**orm***.
A subject called "Performance" would then have been swallowed by a
subject called "ORM". Every pattern here compiles with word boundaries
and is asserted against that case.

**Order is significance.** Patterns are tried most-specific-first within a
subject, so "Spark SQL joins" lands under SQL→Joins rather than
Spark→Performance. Where a label genuinely spans two subjects the first
match in the declared order wins, and that order is written down here
rather than emerging from dictionary iteration.

**Nothing is invented.** A label that matches nothing is not dropped and
does not get a fabricated topic: it lands in its subject's ``general``
subtopic and is counted, so the residual is visible instead of hidden
inside a tidy-looking navigation bar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _stem(word: str, *, final: bool) -> str:
    """
    One word of a phrase, matched the way English actually inflects it.

    The tail is permissive -- ``\\w*`` -- because that is what lets a
    subtopic keyword written in the singular match the plural and the
    gerund a corpus actually uses. Two attempts at a stricter inflection
    set were made and both were worse: they closed the tail to a listed
    set of endings and lost 130 real labels, including "Auto Loader",
    "Column selection" and "Amazon Leadership Principles", because those
    matched for reasons the closure had not anticipated.

    What a plain ``\\w*`` cannot do is the irregular English forms that
    make up a large share of this corpus:

        warehouse -> warehousing   the ``e`` is dropped
        query     -> queries       the ``y`` becomes ``ies``
        technology-> technologies  the ``y`` becomes ``ies``

    Measured consequence of not handling them: ``_word("data warehouse")``
    matched "data warehouse" and not "data warehousing", so a label naming
    the very subject a candidate revises under fell through to the
    catch-all. "Data warehousing" appears on eight posts carrying real
    warehouse questions and was filed nowhere.

    So the irregular stems are *added* to the permissive tail rather than
    substituted for it. Only a word ending in ``e`` or ``y`` gets the
    extra branch, which is also where the false-positive risk lives --
    ``file`` must not reach "filter", and it does not, because the tail
    still starts from the whole word.
    """

    escaped = re.escape(word)

    if not final:
        # Only a plural, and no tail. A middle word of a phrase is never
        # inflected enough to need more, and "data" must not match
        # "database".
        return rf"{escaped}s?(?!\w)"

    if word.endswith("e"):
        # Two branches, and the distinction between them matters.
        #
        # The first keeps the whole word with its open tail: "warehouse",
        # "warehouses", and anything else built on it.
        #
        # The second drops the ``e`` and takes only a listed inflection:
        # that is the only way to reach "warehousing", where the ``e``
        # is not there. It is deliberately closed, because an open tail
        # on the shortened stem reaches "filter" and "film" from "file" --
        # which a keyword list for a data course must not do.
        return (
            rf"(?:{escaped}\w*|{re.escape(word[:-1])}"
            r"(?:s|es|ed|ing|d)?)(?!\w)"
        )

    if word.endswith("y"):
        # "queries" is "quer" plus "ies", and "quer" alone also reaches
        # "quarter", so the ``y`` forms are listed rather than left to a
        # tail.
        return (
            rf"(?:{escaped}\w*|{re.escape(word[:-1])}"
            r"(?:ies|s|es|ed|ing|d)?)(?!\w)"
        )

    return rf"{escaped}\w*(?!\w)"


def _word(*alternatives: str) -> re.Pattern[str]:
    """
    Match any of several phrases, on word boundaries.

    Each argument is one alternative and may itself be several words, so
    ``_word("inner join", "broadcast")`` means "inner join" **or**
    "broadcast". Alternatives are the point: almost every subtopic lists
    a dozen unrelated terms.

    Within one phrase the words must appear in order, separated by a
    space, hyphen or underscore, so "data type" matches "data types" and
    "data-types" but not "data transformation".

    Two details are load-bearing:

    * **The inflection set**, via :func:`_stem`. See there for the
      irregular forms that a plain suffix misses.
    * **``\\b`` at the start.** An unanchored sweep for ``orm`` matched
      116 real labels, every one of them inside "perf**orm**ance",
      "plat**orm***" or "transf**orm***".
    """

    branches: list[str] = []

    for alternative in alternatives:
        words = alternative.split()

        parts = [
            _stem(word, final=position == len(words) - 1)
            for position, word in enumerate(words)
        ]

        branches.append(r"[\s_-]+".join(parts))

    return re.compile(r"\b(?:" + "|".join(branches) + r")", re.IGNORECASE)


@dataclass(frozen=True)
class Subtopic:
    """One revisable subject area inside a major topic."""

    name: str
    patterns: tuple[re.Pattern[str], ...] = ()

    @property
    def slug(self) -> str:
        return _slug(self.name)


@dataclass(frozen=True)
class Major:
    """One major subject, and everything that belongs to it."""

    name: str
    blurb: str
    subtopics: tuple[Subtopic, ...]
    patterns: tuple[re.Pattern[str], ...] = ()

    @property
    def slug(self) -> str:
        return _slug(self.name)

    def general(self) -> Subtopic:
        """The fallback subtopic, for labels nothing else claimed."""

        return self.subtopics[-1]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


# ---------------------------------------------------------------------
# The subjects
# ---------------------------------------------------------------------
#
# Declared in the order a reader should meet them: the languages first
# because they are the most revised, then the platforms, then the
# general practice, then the non-technical material last because it is
# not what most visits are for.


def _sql() -> Major:
    return Major(
        name="SQL",
        blurb="Querying relational data: joins, windows, aggregation and "
              "the optimisation work that follows.",
        patterns=(
            # "postgres", "postgresql", "mysql" and "mssql" were here
            # and are gone. They named *databases*, not the subject:
            # "In a MySQL-to-PostgreSQL CDC flow using Debezium and
            # Kafka, what failure mode and verification ..." is a change
            # data capture question and was being filed under SQL, and
            # "moving MySQL changes through a CDC pipeline" never
            # reached the CDC subtopic because the engine name won the
            # subject. A question about SQL says SQL.
            _word("sql", "sqlite", "plsql", "tsql", "bigquery"),
        ),
        subtopics=(
            Subtopic("Joins", (
                _word("join", "inner join", "outer join", "left join",
                      "right join", "full join", "cross join", "self join",
                      "anti join", "semi join"),
            )),
            Subtopic("Window Functions", (
                # "How would you use SQL window logic to add a
                # previous-month sales difference for ev...", "How would
                # you calculate YOY and MOM growth in SQL?", "How would
                # you remove duplicates by name in PySpark using
                # ROW_NUMBER()?" -- period-over-period comparison, which
                # is a window-function job whatever it is called.
                _word("window logic", "previous-month", "previous month",
                      "year-over-year", "yoy", "mom growth",
                      "month-over-month", "month over month",
                      "sales difference", "period-over-period",
                      "row_number()", "running total", "cumulative sum",
                      "window function", "second-highest", "second highest",
                      "nth highest", "top n per group", "row_number", "rank", "dense_rank",
                      "ntile", "lag", "lead", "first_value", "last_value",
                      "over clause", "partition by", "running total",
                      "cumulative"),
            )),
            Subtopic("CTEs and Subqueries", (
                # "Walk through how the recursive query turns the
                # Subsidiary hierarchy into co-investor ..." -- recursion
                # is the whole point of a recursive CTE.
                # "subqueries" is not "subquery" plus a suffix -- it is
                # "subquer" plus "ies" -- so the plural is listed rather
                # than left to a stem, which is why this subtopic was
                # empty while the corpus clearly had subquery questions.
                _word("cte", "common table expression", "subquery",
                      "subqueries", "sub query", "sub queries",
                      "recursive cte", "with clause",
                      "nested query", "derived table",
                      "correlated subquery", "uncorrelated subquery",
                      "exists clause", "inline view",
                      # "Walk through how the recursive query turns the
                      # Subsidiary hierarchy into co-investor ..." --
                      # recursion is the whole point of a recursive CTE.
                      "recursive query", "hierarchical query",
                      "turns the subsidiary", "co-investor",
                      "hierarchy into", "walk through how the recursive"),
            )),
            Subtopic("Filtering and Set Operations", (
                _word("where clause", "order by", "group by", "having",
                      "union", "union all", "intersect", "except",
                      "distinct", "limit clause", "top n", "case when",
                      "null handling", "coalesce", "is null",
                      "filtering", "predicate", "in list",
                      "between operator", "not between", "between and",
                      # "What are the BETWEEN, IN, and LIKE operators in
                      # SQL?"
                      "between, in, and like", "in and like",
                      "operators in sql",
                      # "When does the post recommend using EXISTS
                      # instead of IN in a SQL query?"
                      "exists", "instead of in", "not in",
                      "semi-join predicate"),
            )),
            Subtopic("Per-Group Ranking and Top-N", (
                # "How do I filter the top 3 products by revenue within
                # each category?", "assign IDs to records ... then
                # retrieve the odd-", "detect circular references in a
                # hierarchical table structure", "find gaps in a
                # sequence, such as missing invoice numbers", "update
                # data in one table based on values from another", "update
                # data in one table".
                #
                # Recurring shapes that are not aggregation and not
                # window functions but behave like both: they are all
                # "rank or relate rows within a group", which is the
                # thing a candidate is actually being asked to do.
                _word("within each", "for each category", "each category",
                      "per category", "per department", "per region",
                      "odd-", "even-numbered", "assign id", "assign ids",
                      "row number without", "sequence such as",
                      "gaps in a sequence", "missing invoice",
                      "circular reference", "hierarchical table",
                      "self-referencing", "update data in one table",
                      "based on values from another",
                      "update one table from", "join a table to itself",
                      "row-level update"),
            )),
            Subtopic("Aggregation and GROUP BY", (
                # "How do you calculate the median salary in a
                # department?", "Find employees earning more than their
                # department average", "Give a SQL query to retrieve the
                # nth record", "most experienced employees in each
                # project" -- per-group statistics, which is what the
                # existing group-by keywords did not reach.
                _word("group by", "aggregate", "aggregation", "having",
                      "count", "sum", "avg", "min", "max", "rollup",
                      "cube", "grouping set", "distinct", "median",
                      "nth record", "nth highest", "per group",
                      "department average", "average salary",
                      "percentile", "standard deviation", "variance",
                      "moving average", "running total", "cumulative sum",
                      "first order", "last order", "consecutive",
                      "top n", "most frequently", "least frequently",
                      "returning user", "retention", "cohort",
                      # Bare "duplicate" was here and it cost a subject:
                      # it matched "a daily cron job to Airflow while
                      # addressing duplicate processing" on length and
                      # sent the question to SQL. The phrase, not the
                      # word.
                      "duplicate record", "duplicat record",
                      "duplicate removal", "remove duplicate",
                      "deduplicat", "latest timestamp"),
            )),
            Subtopic("Query Writing and Problem Solving", (
                # The largest single group in SQL/General: "Write a SQL
                # query to ...", "How should a candidate approach a SQL
                # problem before writing the query", "formulate and
                # verify a correct query", "How would you use SQL to
                # clean and shape incoming log data?". A candidate
                # revises this as a unit -- it is how you decompose an
                # under-specified question -- and it is not the same
                # skill as tuning a query someone else wrote.
                _word("write an sql query", "write an sql expression",
                      "write a sql expression",
                      "write a sql query", "sql query to", "query to",
                      "give a sql query", "approach a sql problem",
                      "formulate", "formulating", "verify a correct",
                      "practical sql problem", "assessment",
                      "under-specified", "decompose the problem",
                      "intermediate sql", "advanced sql",
                      "what would you write", "write a query",
                      "answer a question", "puzzle",
                      "use sql to clean", "use sql to shape",
                      "shape incoming", "write sql to calculate",
                      "write sql to find", "using sql",
                      "begin solving the problem"),
            )),
            Subtopic("Transactions and Locking", (
                # "Explain SQL concurrency and locking, including
                # deadlocks and isolation levels", "How can you detect
                # and prevent deadlocks". Its own subject: the
                # vocabulary is distinct from indexing and from query
                # structure.
                _word("transaction", "concurrency", "concurrent",
                      "locking", "lock", "deadlock", "isolation level",
                      "read committed", "repeatable read", "serializable",
                      "snapshot isolation", "atomicity", "acid",
                      "commit", "rollback", "all-or-nothing",
                      "write conflict", "lost update"),
            )),
            Subtopic("PIVOT and UNPIVOT", (
                # "How can you transpose columns into rows, or unpivot
                # them?", "How do you pivot rows into columns". A
                # specific, frequently-asked, revisable skill.
                _word("pivot", "unpivot", "transpose", "cross tab",
                      "crosstab", "rows into columns",
                      "columns into rows"),
            )),
            Subtopic("SQL Internals and Execution", (
                # "Explain the end-to-end path from a SQL statement to
                # returned query results", the ANSI SQL-mode casting
                # difference. What happens between the statement and the
                # result set.
                _word("query execution", "execution plan", "query plan",
                      "parser", "parse tree", "end-to-end path",
                      "from a sql statement to", "ansi sql",
                      "sql mode", "casting difference", "cast",
                      "implicit cast", "type conversion",
                      "collation", "execution engine",
                      "how a query is executed", "set operators",
                      # "What is the logical execution order of the
                      # clauses ...", "What stages does a database
                      # system use to process and execute a SQL
                      # statement?"
                      "logical execution order", "execution order",
                      # "How does a SQL statement become an executable
                      # plan?", "What steps does a database system follow
                      # when executing a SQL statement?", "How does an
                      # SQL query move through the database subsystems
                      # from arrival to the collectio..."
                      "executable plan", "become an executable",
                      "steps does a database system follow",
                      "database subsystems",
                      "move through the database",
                      "logical-order diagram",
                      "visible sql clauses",
                      "clause order", "stages does a database",
                      "process and execute a sql",
                      "end-to-end path in sql"),
            )),
            Subtopic("SQL Fundamentals", (
                # "What is SQL?", "why is SQL important for aspiring data
                # engineers?", "What SQL knowledge should a data analyst
                # prioritize", "what SQL skills does the post emphasize
                # beyond writing queries?". Definitional and scope
                # questions, asked constantly at the start of an
                # interview and answered nowhere else.
                _word("what is sql", "why is sql important",
                      "sql skills", "sql knowledge",
                      "importance of sql", "role of sql",
                      "sql in a data engineering pipeline",
                      "sql across the stages", "sql operations",
                      "purpose of sql", "sql is a",
                      "sql as a foundation", "sql fundamentals",
                      "basics of sql", "introduction to sql",
                      "help integrate applications with databases",
                      "clauses form the basic starting point",
                      "sql command categories",
                      # "How does the post describe SQL's role across the
                      # stages of a data engineering pipeline?", "Why does
                      # the author argue that data engineers still need SQL
                      # when using a modern data platform?"
                      "sql's role", "still need sql",
                      # "Which SQL capabilities are prioritized in Week 1
                      # of the roadmap?", "Which SQL capabilities does
                      # the post associate with this analysis ...?",
                      # "Which SQL techniques does the post identify as
                      # important in layered data transformations ...?",
                      # "Which recurring SQL topics does the post
                      # identify as the practical core ..."
                      "which sql capabilities", "which sql techniques",
                      "which sql constructs", "which sql features",
                      "which sql topics", "which sql-related",
                      "which sql-related features", "recurring sql topics",
                      "practical core",
                      "learning sql", "challenge does the post identify in learning",
                      "real-project sql", "basic crud",
                      "foundational skill", "syntax alone",
                      "sql syntax alone", "efficient sql",
                      "categories of sql", "named sql operation",
                      "extracting the domain",
                      "exchange data",
                      "applications and database systems",
                      # "A candidate needs to test SQL queries quickly
                      # without following a fixed course.", "Where does
                      # the post recommend practicing SQL ...?", "Which
                      # resources from the post are most suitable for a
                      # beginner ...?"
                      "test sql queries quickly", "practicing sql",
                      "practice sql", "beginner who wants structured",
                      "case-study-style sql",
                      # "What analytical stages are required by the SQL
                      # problem described in the post?"
                      "analytical stages", "problem described in the post",
                      # "How would you implement SCD Type 2 logic in SQL?"
                      # "How would you structure a compliant SQL query"
                      # for the stated schema?"
                      "scd type 2 logic in sql", "compliant sql query",
                      "role of sql in a", "sql in the pipeline",
                      # "What is a database?"
                      "what is a database",
                      "sql support data analysis and reporting",
                      "business questions were addressed",
                      "airline sentiment pipeline",
                      "magic tables", "sql pattern to reach for",
                      "sql clauses form"),
            )),
            Subtopic("SQL Against External Data", (
                # "How can SQL be used to ingest data from a relational
                # database in the big data environment?", "What
                # SQL-based storage and processing examples are
                # presented ...?", "According to the roadmap, which
                # components are intended for SQL-based querying of S3"
                _word("sql-based", "querying of s3", "querying s3",
                      "big data environment", "sql on data lake",
                      "external table", "federated query",
                      "ingest data from a relational database"),
            )),
            Subtopic("String, Date and Numeric Functions", (
                # "different ways to fetch the first five characters of a
                # string in SQL", "How do DATE_ADD and DATE_SUB perform
                # date arithmetic"
                # "Write a SQL expression that converts delivery_time to
                # its date with the time set to 00:0...", "Is order 4
                # delayed when order_time is 16:00:00, deliver_time is
                # 17:30:00 ..."
                _word("converts delivery_time", "time set to 00",
                      "order_time", "deliver_time",
                      "date_format", "date_trunc", "date_add", "date_sub",
                      "date arithmetic",
                      "date functions", "datediff", "extract year",
                      "first five characters", "string functions",
                      "substring", "left(", "trim(", "coalesce",
                      "greatest and least", "floor and ceil",
                      "rounding", "number formatting"),
            )),
            Subtopic("SQL and NoSQL Choice", (
                # "When would SQL be preferable to NoSQL, and when would
                # NoSQL be considered?"
                _word("sql vs nosql", "preferable to nosql", "nosql",
                      "relational vs", "rdbms", "document database",
                      "which database to choose", "polyglot"),
            )),
            Subtopic("Query Reading and Debugging", (
                # A distinct group: "In the example query, what status
                # description is returned for A, I, S ...", "What ename
                # and orgname values does the query derive for employee
                # 103?", "Why does the query pass CHARINDEX('@', emailid)
                # as the third argument ...", "What query conditions are
                # required to calculate the result correctly ...".
                #
                # Reading SQL written by someone else, working out what it
                # does and whether it is right, is a skill candidates are
                # asked for and cannot revise from the "how do I write
                # one" material.
                _word("the example query", "what does the query",
                      "the query derives", "query conditions are required",
                      "why does the query", "the supplied query",
                      "expected flags", "sample data",
                      "status description is returned",
                      "ename and orgname", "sales_diff",
                      "rule produces a negative",
                      "which columns in", "columns in order_details",
                      "information is missing from the prompt",
                      "calculation and query restrictions",
                      "qualifying playlists",
                      "common to both examples",
                      "can responsibly be drawn",
                      # "What company-level comparison must the SQL
                      # query return to satisfy the prompt?", "How are
                      # promotion-acquired inactive customers ...
                      # organised?", "What origin and final destination
                      # pairs are shown for the two customers in the
                      # sample Fl...", "What output does the Flipkart
                      # exercise require?"
                      "satisfy the prompt", "company-level comparison",
                      "promotion-acquired", "promotion-dependent",
                      "flipkart exercise", "origin and final destination",
                      "clarification is needed", "clarified before",
                      "implementation details should be",
                      "unresolved semantics", "should be clarified",
                      "how does the exercise distinguish",
                      "the query omits", "what clarification is needed",
                      "which fields shown in",
                      "the query return only", "read the query",
                      "walk through the query",
                      "what the query does", "debug the query",
                      "query returns the expected"),
            )),
            Subtopic("SQL Security", (
                # "How do you prevent SQL injection attacks in your
                # queries?"
                _word("sql injection", "injection attack", "parameterized",
                      "parameterised", "prepared statement",
                      "sanitize input", "sanitise input",
                      "least privilege", "grant", "revoke"),
            )),
            Subtopic("Query Optimization", (
                _word("using limit", "limit to find", "poor choice for finding",
                      "limit clause in the", "highest employee",
                      "ways to optimize sql", "optimize sql queries",
                      "query appears to hang", "query is slow",
                      "slow in production", "anti-pattern",
                      "antipattern", "report performs poorly",
                      # "How would you optimize a slow SQL query?", "How
                      # would you optimize slow SQL queries operating on
                      # large datasets?" -- the most common shape in
                      # SQL/General by a distance.
                      "optimize a slow sql", "optimize slow sql",
                      # "How would you optimize a slow-running SQL
                      # query?", "How do Parquet and ORC improve SQL scan
                      # efficiency, and what file-layout pitfall can und...",
                      # "Which resource metric spikes during the MSSQL
                      # Server slowdown ...?", "How do you perform
                      # time-based partitioning in a SQL database?"
                      "slow-running sql", "slow running sql",
                      "parquet and orc", "scan efficiency",
                      "file-layout", "file layout",
                      "mssql server slowdown", "resource metric spikes",
                      "time-based partitioning",
                      "partitioning in a sql",
                      "optimise a slow sql", "slow sql query",
                      "investigate and improve the performance",
                      "performance of a slow",
                      "query optimization", "query optimisation",
                      "index", "indexing", "execution plan", "explain plan",
                      "query plan", "cost based optimizer", "statistics",
                      "slow query", "n+1",
                      # "A SQL query returns the expected products when
                      # there is one top-selling category, but it ...", "How
                      # do you optimize a slow-performing SQL query",
                      # "restructured to improve readability and
                      # maintainability", "loads data for a dashboard".
                      "slow-performing", "slowly performing",
                      "optimize a sql", "optimise a sql",
                      "tuning", "bottleneck", "readability",
                      "maintainability", "modularity", "rewrite the query",
                      "refactor", "incorrect result", "wrong result",
                      "expected result", "returns the expected",
                      "sargable", "composite index", "covering index",
                      "full table scan", "table scan"),
            )),
            Subtopic("Views and Stored Procedures", (
                _word("view", "materialized view", "stored procedure",
                      "trigger", "function", "procedure", "cursor",
                      "temporary table", "temp table",
                      # "Which SQL features does the post identify for
                      # reusable database logic and managing d..."
                      "reusable database logic",
                      "managing data in sql", "task automation",
                      "automated responses",
                      "table creation, schema modification"),
            )),
            Subtopic("Data Types and Constraints", (
                _word("data type", "primary key", "foreign key",
                      "constraint", "normalization", "normalisation",
                      "1nf", "2nf", "3nf", "null", "duplicate key",
                      "schema design",
                      # "Explain Auto Increment in SQL.", "IDENTITY
                      # behavior after deletion and truncation", "How do
                      # you audit changes, including INSERT, UPDATE, and
                      # DELETE", "denormalization".
                      "auto increment", "identity column", "identity",
                      "auto_increment", "serial", "surrogate key",
                      "denormalization", "denormalisation",
                      "insert", "update statement", "delete statement",
                      "truncate", "drop table", "create table",
                      "alter table", "dml", "ddl", "audit changes",
                      "change tracking", "soft delete", "temporal table",
                      # "Write the SQL command shown in the post to create
                      # a shallow clone of prod.sales_gold nam...", "What
                      # data-modification and database-management
                      # capabilities does the post attribute to S..."
                      "create a shallow clone", "clone of",
                      "data-modification", "database-management",
                      "scd type 2", "history tracking"),
            )),
            # Removed: "SQL in Spark". It was declared and then held
            # nothing. Eleven questions in the corpus are about SQL in
            # Spark -- "What is the difference between RANK() and
            # DENSE_RANK() in Spark SQL?", "How would you use Spark SQL to
            # retrieve the most recent transaction for each customer?" --
            # and every one of them is placed under Spark / Spark SQL,
            # which is where a candidate revises them. Declaring a second
            # home for the same questions would have produced a duplicate
            # in the revision index and then measured as a dead end.
            Subtopic("General", ()),
        ),
    )


def _spark() -> Major:
    return Major(
        name="Spark and PySpark",
        blurb="Distributed execution: DataFrames, when work actually "
              "runs, and why a job is slow.",
        patterns=(
            _word("spark", "pyspark", "rdd", "sparksession",
                  "dataframe", "spark sql"),
        ),
        subtopics=(
            Subtopic("Joins", (
                _word("broadcast join", "broadcast hash join", "shuffle join",
                      "sort merge join", "spark join", "merge join",
                      # "How do you perform a left anti join and left semi
                      # join in PySpark SQL?", "How can a join with a small
                      # table be optimized in PySpark?", "What types of
                      # joins does Spark support, and why is broadcast
                      # required?"
                      # "When does the post recommend using
                      # broadcast(df), and what benefit does it claim?",
                      # "Which Spark-specific join does the post
                      # identify as an alternative in its discussion o..."
                      "broadcast(df)", "using broadcast",
                      "spark-specific join",
                      "alternative in its discussion",
                      "anti join", "semi join", "left semi",
                      "left anti", "outer join", "inner join", "cross join",
                      "full outer", "self join", "equi join",
                      "non-equi", "splat join", "join condition",
                      "join with a small table", "join strategy",
                      "types of joins", "why is broadcast required",
                      "broadcast required", "conditional joins"),
            )),
            Subtopic("Partitions and Partitioning", (
                _word("partition", "partitioning", "repartition",
                      "repartitioning", "coalesce", "skew"),
            )),
            Subtopic("Shuffle and Performance", (
                # "When does Spark fail to push down filters, and how can
                # that be addressed?", "What strategies do you use for
                # optimizing Spark jobs?", "What processing change
                # reduced the described Spark job from 55-60 minutes to 28
                # minutes ...?"
                # "What processing change reduced the described Spark
                # job from 55-60 minutes to 28 minutes ...?"
                _word("minutes to 28", "processing change reduced",
                      "push down filters", "filter pushdown",
                      "predicate pushdown", "pushdown", "push down",
                      "strategies do you use for optimizing",
                      "optimizing spark job", "reduce the described spark job",
                      "shuffle", "performance", "optimization",
                      "optimisation", "tuning", "slow", "bottleneck",
                      "spill", "adaptive query execution", "aqe",
                      "skew join", "data skew", "salting", "sharding",
                      "predicate pushdown", "columnar storage",
                      "hot key", "repartition"),
            )),
            Subtopic("Structured Streaming", (
                _word("streaming", "watermark", "late-arriving",
                      "late arriving", "late data", "exactly-once",
                      "idempotency", "idempotent", "checkpoint",
                      "event time", "micro-batch", "microbatch"),
            )),
            Subtopic("Transformations and Actions", (
                # "What are the Extract, Transform, and Load stages in
                # the basic PySpark ETL workflow", "Build a PySpark
                # pipeline that cleans, transforms, and aggregates data
                # for reporting", "A PySpark join unexpectedly produces
                # many more rows than expected".
                # "What are the three stages of ETL, and what happens
                # during each stage in a PySpark work..."
                _word("three stages of etl", "stages of etl",
                      "etl workflow", "extract, transform, and load",
                      "clean, transform, and aggregate",
                      "pipeline that cleans", "unexpectedly produces",
                      "more rows than expected", "transformation", "action",
                      "lazy evaluation",
                      "laziness", "narrow dependency", "wide dependency",
                      "d ag", "groupby", "groupbykey", "reducebykey",
                      "foreach", "collect", "count action"),
            )),
            Subtopic("DataFrames and RDDs", (
                # "Which PySpark functions does the post identify for
                # date arithmetic and extracting date...", "Which
                # PySpark functions does the post place in the
                # column-operation and aggregation ca...", "Which
                # PySpark method does the post identify for using
                # SQL-style expressions?", "What operations does the
                # provided PySpark example perform?"
                _word("pyspark functions", "pyspark methods",
                      "pyspark operations", "pyspark feature",
                      "pyspark method", "pyspark function",
                      "provided pyspark example",
                      "pyspark storage and reading",
                      # "How are aggregations performed in PySpark, and which
                # functions and methods are commonly used?" -- and the
                # pipe-delimited CSV and nested-JSON reading questions.
                      "aggregations are performed in pyspark",
                      "pipe-delimited csv", "pipe delimited csv",
                      "dynamic nested json", "nested json",
                      "how do you read a text file",
                      "file formats supported by pyspark",
                      "formats supported by pyspark sql",
                      "alias()", "renaming a pyspark column",
                      "current_date()", "current_timestamp()",
                      "withcolumn()", "aggregations are performed in pyspark",
                      "dataframe", "rdd", "dataset", "spark session",
                      "sparkcontext", "schema", "select", "with column",
                      "lit", "udf", "pandas udf",
                      # "How do you read and write CSV, Parquet, and JSON
                      # data in PySpark?", "How are aggregations performed
                      # in PySpark, and which functions ...", "How can a
                      # map column be exploded and flattened", "string
                      # column be converted to timestamps", "Define JDBC
                      # and explain how it's used in connecting Spark to
                      # databases", "Kryo serialization differ from Java
                      # serialization".
                      "read csv", "write csv", "read parquet",
                      "write parquet", "read json", "write json",
                      "read and write", "serialization", "kryo",
                      "jdbc", "explode", "explode_outer", "map column",
                      "flatten", "null handling", "coalesce null",
                      "badrecord", "corrupt record", "schema inference",
                      "udf", "user defined function", "groupby",
                      "group by", "agg function", "aggregation function",
                      "collect_list", "collect_set", "when otherwise",
                      "otherwise clause", "string to timestamp",
                      "malformed", "trim", "regexp", "col functions"),
            )),
            Subtopic("Architecture and Execution", (
                # "Explain the architecture of Apache Spark, including
                # its key components", "the driver, cluster manager, and
                # worker", "the Spark execution flow from Job to Stage to
                # Task", "the full lifecycle of a Spark job from
                # submitted code through DAG formation", "how is code
                # written in PySpark executed when Spark is written in
                # Scala", "data locality".
                #
                # Its own subtopic because it is a distinct thing to
                # revise: nothing about the DataFrame API tells you how a
                # job becomes stages.
                _word("spark architecture", "architecture of apache spark",
                      "architecture of spark", "key components",
                      # "How does Apache Spark work under the hood, "
                      # "including its architecture" -- Spark's own
                      # words for it.
                      "how does apache spark work", "under the hood",
                      "how spark works",
                      "cluster manager", "driver", "executor", "worker node",
                      "job to stage", "stage to task", "dag", "lineage",
                      "data locality", "lifecycle of a spark job",
                      "spark application", "driver program",
                      "speculative execution", "task scheduler",
                      "scheduler", "executor memory", "written in scala",
                      # "Which microservices, authentication, and Spark
                      # internals topics were covered in the se...", "Which
                      # three components are explicitly named for Spark
                      # execution?"
                      "spark internals", "microservices",
                      "three components",
                      "components are explicitly named",
                      # "What are broadcast variables in PySpark, and
                      # when would you use them?"
                      "broadcast variable", "broadcast variables",
                      "shared read-only", "accumulators"),
            )),
            Subtopic("Deployment and Operations", (
                # "How do you deploy PySpark applications in production?",
                # "How do you efficiently process a CSV file larger than
                # 10 GB in Spark?", "How does Spark handle memory
                # management, and what are common memory-related
                # pitfalls?", "How would you deploy a PySpark job on a
                # cluster?", "How would you implement an incremental
                # PySpark pipeline that processes new daily d..."
                _word("deploy pyspark", "deployment", "production pyspark",
                      "deploy a pyspark job", "pyspark job on a cluster",
                      "incremental pyspark", "read dynamic nested json",
                      "nested json files", "top n most frequent",
                      "remove duplicates by",
                      "in production", "spark submit", "spark-submit",
                      "cluster size", "scaling up", "memory management",
                      "memory-related", "out of memory", "oom",
                      "large file", "10 gb", "big file",
                      "efficiently process", "chunked", "partitioning strategy"),
            )),
            Subtopic("Spark Fundamentals", (
                # "What is PySpark, and how is it different from Apache
                # Spark?", "What is a task in Spark?", "What is the role
                # of SparkSession in PySpark?", "What is the difference
                # between local mode and cluster mode in Spark?", "Which
                # Apache Spark execution and evaluation fundamentals did
                # the author learn?"
                #
                # Definitional questions, asked of anyone using the
                # framework for the first time and answered nowhere else.
                _word("what is pyspark", "how is it different from apache spark",
                      "what is a task in spark", "role of sparksession",
                      "sparksession", "local mode and cluster mode",
                      "execution and evaluation fundamentals",
                      "how does spark handle big-data",
                      "spark fundamental", "what is spark",
                      "spark context", "sparkconf", "sc.parallelize",
                      "rdd fundamentals"),
            )),
            Subtopic("Window Functions in Spark", (
                # "How would you construct the PySpark window
                # specifications and side-by-side ranking com..."
                _word("window specifications", "side-by-side ranking",
                      "ranking comparison",
                      # "What are window functions in SQL and PySpark,
                      # what types exist, and what use cases do they
                      # serve?" -- the same question SQL's own Window
                      # Functions subtopic holds, asked on Spark.
                      "window functions in sql and pyspark",
                      # "Use a PySpark window function to obtain the top three
                # orders for each customer", "What analytical tasks can
                # PySpark window functions perform without collapsing the
                # data ...", "What are window functions in SQL and
                # PySpark, what types exist, and what use cases do ..."
                #
                # SQL has its own Window Functions subtopic. Spark needs
                # its own because the API, the partitioning and the
                # ordering are all expressed differently, and a candidate
                # is asked for the Spark version separately.
                      "over clause", "partitionby", "rowsbetween",
                      "without collapsing", "analytical tasks",
                      "top three orders", "top n per partition"),
            )),
            Subtopic("PySpark Coding Problems", (
                # "How would you calculate average sales per product
                # category in PySpark?", "How would you use PySpark to
                # find the top N most frequent words in a large text
                # file", "How would you read a text file in PySpark",
                # "How would you implement the requested calculation as
                # a PySpark program", "How would you handle duplicate
                # records in PySpark".
                #
                # The technology is named in the question and nothing
                # matched it. "in pyspark" and "a pyspark program" are
                # the phrases, not a general PySpark pattern, because a
                # general one would take architecture questions too.
                _word("in pyspark", "a pyspark program", "pyspark program",
                      "using pyspark", "pyspark job", "a pyspark job",
                      "spark job that", "a spark job", "in spark",
                      "with pyspark", "pyspark solution",
                      # The imperative form is the largest group in this
                      # subtopic and was missing entirely: "Write a
                      # PySpark query to filter employees where age is
                      # greater tha...", "Write a PySpark script that
                      # replaces missing values in the price column with
                      # the mean ...", "Write a PySpark function to
                      # extract the domain from an email column", "Write
                      # Spark code to read a file from a specified
                      # location."
                      "write a pyspark query", "write a pyspark script",
                      "write a pyspark function", "write a pyspark example",
                      "write a pyspark program", "write pyspark code",
                      "write sql and pyspark",
                      "write spark code", "write pyspark",
                      "use pyspark to calculate",
                      "use pyspark to identify",
                      "withcolumn", "row_number()",
                      "count distinct",
                      "group by and count", "rank within"),
                # "Given a log file, write PySpark code to find the top
                # ten IP addresses by request count", "How can PySpark
                # identify the repeated numbers in 12327738282888?",
                # "Implement a PySpark task that finds product names
                # whose latest month's sales are greater ...", "write
                # queries in both SQL and PySpark".
                #
                # Applied PySpark, as opposed to asking what an API does.
                # A distinct skill and a distinct revision list.
                _word("write pyspark and sql", "pyspark and sql logic",
                      "write pyspark code", "write queries in both",
                        "both sql and pyspark", "oracle sql and pyspark",
                        "pyspark task", "implement a pyspark",
                        "pyspark code to", "log file", "ip addresses by request",
                        "repeated numbers", "latest month's sales",
                        "analytical grain", "what analytical tasks",
                        "rewrite the reported pyspark", "pyspark problem"),
            )),
            Subtopic("Configuration and Sizing", (
                # "What factors should be considered when applying the
                # post's Spark configuration recommendations ...", "What
                # information should be gathered before choosing Spark
                # configurations for a 1 TB file ...", "What Spark 3.4.0
                # capability can reduce boilerplate in PySpark code that
                # uses spark.sql", "How would you optimize a PySpark
                # job?", "How would you optimize a Spark job that merges
                # multiple small Parquet files?"
                _word("optimize a pyspark", "optimize a spark job",
                      "merges multiple small parquet",
                      "small parquet files", "compaction",
                      "spark configuration", "config recommendations",
                      "choosing spark", "spark settings", "sizing",
                      "tune the spark", "spark tuning options",
                      "reduce boilerplate", "spark 3.4", "adaptive query",
                      "spark submit options", "cluster configuration",
                      "1 tb", "spark properties"),
            )),
            Subtopic("Spark Versus MapReduce", (
                # "What are the real-world differences between Spark and
                # MapReduce?", "What are the key differences between
                # Spark 1.6 and Spark 2.x?", "What are the different
                # execution modes in Apache Spark: local, standalone,
                # YARN, and ..."
                _word("spark and mapreduce", "mapreduce", "spark 1.6",
                      "spark 2.x", "execution modes in apache spark",
                      "local, standalone", "standalone cluster",
                      "yarn", "key differences between spark",
                      "version differences", "hadoop vs spark"),
            )),
            Subtopic("Spark Projects and Experience", (
                # "What does the post establish about the Netflix data
                # analytics project, and what implementation ...?", "What
                # aspects of a previous Spark project should a candidate
                # be prepared to explain ...?"
                # "Where exactly did you apply Apache Spark in your past
                # projects?", "What Spark-related issue have you
                # troubleshot, and how did you resolve it?"
                _word("past projects", "where did you apply",
                      "spark-related issue", "troubleshoot",
                      "netflix", "previous spark project",
                      "spark project", "pyspark project",
                      "aspects of a previous", "prepared to explain",
                      "your spark experience", "spark experience",
                      "have you worked with spark"),
            )),
            Subtopic("Ecosystem Integration", (
                # "Why does the post present PySpark as a suitable
                # technology for scalable ETL pipelines?", "Why does the
                # post recommend PySpark instead of Pandas for
                # production-scale ETL?", "When would the post suggest
                # considering PySpark rather than relying only on
                # Pandas?", "How does the post propose running Spark and
                # supporting analytics and storage on GCP?", "Which
                # capabilities and integrations does the post attribute
                # to PySpark?"
                _word("instead of pandas", "rather than relying only on pandas",
                      "scalable etl pipeline", "scalable etl pipelines",
                      "production-scale etl", "capabilities and integrations",
                      "storage on gcp", "spark on gcp",
                      # "How have you integrated PySpark with Hadoop/HDFS, Hive,
                # or Kafka?"
                      "hadoop", "hdfs", "hive", "kafka", "sqoop",
                      "ecosystem", "integrated with", "integration with",
                      "spark sql on hive", "external table",
                      "connecting to", "connect spark"),
            )),
            Subtopic("Caching and Persistence", (
                _word("cache", "caching", "persist", "storage level",
                      "memory_and_disk"),
            )),
            Subtopic("Catalyst and Execution", (
                _word("catalyst", "optimizer", "tungsten", "codegen",
                      "whole stage", "execution plan", "physical plan"),
            )),
            Subtopic("Spark SQL", (
                _word("spark sql", "sparksql", "delta lake", "delta table",
                  "delta format", "delta merge"),
            )),
            Subtopic("General", ()),
        ),
    )


def _databricks() -> Major:
    return Major(
        name="Databricks",
        blurb="Delta Lake, Unity Catalog and the job scheduling around a "
              "lakehouse.",
        patterns=(
            _word("databricks", "unity catalog", "delta lake",
                  "medallion", "lakehouse", "z-ordering", "z ordering",
                  "dbx", "photon", "autoloader", "notebook",
                  "delta lake", "deltalake", "delta transaction",
                "time travel", "liquid clustering"),
        ),
        subtopics=(
            Subtopic("Delta Lake", (
                # "What is SCD Type 2, and how would you implement it in
                # ADF or Databricks?", "What is SCD Type 2, and what steps
                # would you follow to implement SCD Type 2 using Data..."
                _word("what is scd type 2", "scd type 2, and",
                      "implement it in adf", "delta lake", "deltalake",
                      "acid", "transaction log",
                      "time travel", "vacuum", "merge into", "upsert",
                      "z-ordering", "z ordering", "liquid clustering",
                      # "How can Delta Cache be enabled for a frequently
                      # queried Delta table?", "how many records were
                      # inserted or updated in a Delta table", "duplicate
                      # records before writing to a Delta table",
                      # "merge operations to implement SCD Type 2 logic in
                      # Databricks", "optimistic concurrency and
                      # transaction snapshots".
                      "delta cache", "change data feed",
                      "inserted or updated", "duplicate records before",
                      "scd type 2 logic", "scd type 2 in databricks",
                      "optimistic concurrency", "transaction snapshot",
                      "write to a delta", "delta transaction",
                      "constraint", "expectation on a delta"),
            )),
            Subtopic("Unity Catalog", (
                _word("unity catalog", "catalog", "metastore",
                      "governance", "lineage", "permission", "grant"),
            )),
            Subtopic("Workflows and Jobs", (
                # Not bare "job": it matched "cron job" and "Spark job",
                # which belong to Data Engineering and Spark.
                _word("job cluster", "databricks job", "workflow",
                      "notebook", "cluster",
                      "job cluster", "all-purpose", "dbx", "pipeline run",
                      # "According to the post, what problems is the
                      # Databricks quickstart template intended to ...?",
                      # "What hands-on PySpark and Databricks capabilities
                      # does the author report using?", "What operational
                      # and validation requirements are described for
                      # migrating the activitylog ...?"
                      "quickstart template", "quickstart",
                      "author report using", "activitylog",
                      "migrating the", "validation requirements",
                      "promote code from lower", "environments"),
            )),
            Subtopic("Performance", (
                # "How do you optimize a Spark job that is running slowly
                # in Databricks?", "According to the post, which
                # maintenance and tuning tasks are associated with using
                # Da...", "How would you configure Databricks compute to
                # process a 50GB file with Spark?", "Which PySpark or
                # Databricks operations would you use to persist
                # partitioned tables and ..."
                _word("optimize a spark job", "running slowly in databricks",
                      "maintenance and tuning", "databricks compute",
                      "50gb", "persist partitioned",
                      "partitions and",
                      "photon", "performance", "optimization",
                      "optimisation", "small file", "file size",
                      "auto optimize", "liquid"),
            )),
            Subtopic("File Formats", (
                _word("parquet", "orc", "avro", "csv", "json file",
                      "file format", "compression", "snappy",
                      # "How do you read and write CSV, Parquet, and JSON
                      # data in PySpark?" reaches here too, which is why
                      # the read/write verbs are listed rather than the
                      # formats alone.
                      "delta format", "iceberg", "hudi",
                      "shallow clone", "deep clone", "clone",
                      "copy into", "external table"),
            )),
            Subtopic("Databricks Fundamentals", (
                # "What technology is Databricks built on, and what
                # capability does the post associate wi...", "How does
                # the post define Lakebase, and which use cases does it
                # associate with the serv...", "What configuration
                # problem is the all-in-one Databricks template
                # intended to address?", "How do lateral joins expand
                # the possible use cases of Databricks TVFs?", "Why does
                # the post characterize the proposed application
                # architecture as client-native ..."
                _word("lakebase", "all-in-one databricks",
                      "template intended to address", "lateral join",
                      "databricks tvf", "tvfs",
                      "client-native", "application architecture",
                      "databricks is built on",
                      # "What is Databricks, and which major data activities
                      # does the post say it unifies?", "What relationship does
                # the post identify between Databricks and Apache Spark?",
                # "What is the difference between a Data Lake, a
                # Warehouse, and a Lakehouse?", "What does a Lakehouse
                # combine, and which workload modes does it support?"
                      "what is databricks", "databricks and apache spark",
                      "databricks on azure unifies", "what does a lakehouse",
                      "workload modes", "operational responsibilities",
                      "operational responsibility", "unifies",
                      "platform fundamentals", "what is a databricks runtime",
                      "databricks runtime", "databricks apps",
                      "databricks connect", "ai functions",
                      "databricks ai functions"),
            )),
            Subtopic("Auto Loader and Ingestion", (
                # "What is Databricks Auto Loader, and in which scenarios
                # is it most useful?", "What changes when File events are
                # enabled instead of using the legacy notification queue?"
                # "How would you process millions of files in a Data
                # Lake efficiently in batches using AD...", "A PySpark
                # pipeline on Databricks takes more than four hours to
                # discover new files in a ..."
                _word("millions of files", "in batches",
                      "four hours to discover", "discover new files",
                      "auto loader", "autoloader", "file events",
                      "notification queue", "cloudfiles",
                      "incremental ingestion", "landing zone",
                      "file arrival", "new files"),
            )),
            Subtopic("Managed Tables and SQL Analytics", (
                # "What benefits does the post attribute to Databricks
                # Managed Tables?", "What integration problem remains
                # unresolved when querying Databricks Managed Tables ...",
                # "What does the post state about using SQL Server and
                # Databricks together?", "How does the example load
                # cleaned Databricks data into Azure SQL Database?"
                _word("managed table", "managed tables",
                      "sql server and databricks",
                      "databricks and sql server",
                      "azure sql database", "load cleaned databricks",
                      "databricks sql warehouse", "sql warehouse",
                      "databricks and sql analytics",
                      "querying databricks managed"),
            )),
            Subtopic("Lakehouse Architecture", (
                # "How do a data lake, data warehouse, data mart, and
                # data lakehouse differ?", "What sequence does the
                # proposed AI and data system architecture follow?"
                _word("data lake, data warehouse", "data lakehouse",
                      "lakehouse architecture", "data mart",
                      "lake versus warehouse", "lake vs warehouse",
                      "medallion architecture", "bronze, silver, and gold",
                      "data lake versus", "lakehouse concept"),
            )),
            Subtopic("Databricks Projects and Experience", (
                # "Explain a recent project that used Databricks.", "Have
                # you worked with PySpark on Azure Databricks? Explain a
                # use case."
                _word("recent project that used databricks",
                      "project that used databricks",
                      "worked with pyspark on azure databricks",
                      "azure databricks", "your databricks experience",
                      "databricks project", "explain a recent project"),
            )),
            Subtopic("Integration with Azure", (
                # "How do ADF and Databricks differ, and how do they work
                # together?", "How do you connect ADLS Gen2 with Azure
                # Databricks, and where are role assignments set?"
                _word("adf and databricks", "databricks and adf",
                      "connect adls", "adls with azure databricks",
                      "databricks integration", "integrate databricks",
                      "unity catalog with", "role assignments",
                      "databricks and snowflake",
                      "databricks and adls", "azure databricks"),
            )),
            Subtopic("Delta Live Tables and ETL", (
                # "Explain Databricks Delta Live Tables and when you
                # would use batch versus streaming.", "A Lakehouse team
                # wants to streamline ETL while reducing manual effort.",
                # "How does TRIGGER ON UPDATE change the refresh
                # behavior of a Databricks materialized ...?", "What
                # problem does Databricks APPLY CHANGES INTO ... FROM
                # SNAPSHOT address ...?", "What is SCD Type 2 ... using
                # Databricks?"
                _word("delta live tables", "dlt", "live table",
                      "streamline etl", "pipeline as code",
                      "expectations", "dlt pipeline",
                      "trigger on update", "materialized view",
                      "materialised view", "apply changes into",
                      "from snapshot", "scd type 2 in databricks",
                      "refresh behavior", "streaming table",
                      "append-only table"),
            )),
            Subtopic("Platform Comparison and Choice", (
                # "When should Databricks, Spark, or ADF be used?", "Why
                # should ingestion, cleansing, and aggregation logic be
                # separated into different Med..."
                _word("when should databricks", "databricks, spark, or adf",
                      "separated into different", "medallion",
                      # "A team is choosing between Apache Spark and
                # Databricks. What decision criterion ...", "Does the
                # reported skewed-join result establish that Snowflake
                # generally performs better ...", "Describe an ETL
                # pipeline using ADF or Databricks and Snowflake".
                #
                # Its own subtopic: "when would you choose which" is a
                # different question from "how does it work", and it is
                # asked constantly.
                "choosing between", "when to use", "which is better",
                      "compare ", "comparison", "versus ", " vs ",
                      "snowflake", "redshift vs", "bigquery vs",
                      "decision criterion", "trade-off", "tradeoff",
                      "pros and cons"),
            )),
            Subtopic("Schema Evolution", (
                _word("schema evolution", "schema enforcement",
                      "column mapping", "data skipping",
                      # "how does type-casting behavior differ between
                      # Databricks ...", "what ANSI SQL-mode observation is
                      # associated with the casting difference".
                      "type-casting", "type casting", "casting behaviour",
                      "casting behavior", "cast behavior",
                      "overwriteSchema", "mergeSchema",
                      "merge schema", "schema change", "column added",
                      "backward compatibility"),
            )),
            Subtopic("General", ()),
        ),
    )


def _python() -> Major:
    return Major(
        name="Python",
        blurb="The language the pipelines are written in: structures, "
              "functions, and the object model underneath.",
        patterns=(
            _word("python", "pyscript", "pandas", "numpy", "list",
                  "dictionary", "tuple", "decorator", "generator",
                  "iterator", "oop", "exception handling", "lambda",
                  "recursion", "dataclass", "asyncio", "threading",
                  "multiprocessing", "pytest", "type hint",
                  "object oriented"),
        ),
        subtopics=(
            Subtopic("Data Structures", (
                _word("list", "dictionary", "dict", "tuple", "set",
                      "deque", "counter", "frozenset", "array",
                      "data structure", "linked list", "stack", "queue",
                      # "Explain the difference between deepcopy and
                      # shallow copy in Python."
                      "deepcopy", "shallow copy", "copy module",
                      "mutable vs immutable", "reference type"),
            )),
            Subtopic("Files and Large Data", (
                # "How do you handle a file that is too large to fit into
                # memory in Python? Show an example.", "How would you
                # process a 100GB CSV in Python without causing memory
                # issues?", "In Python, how would you optimize data
                # transformations when processing a 10GB CSV fi...",
                # "How would you write Python code that requests data
                # from a public weather API and pro..."
                _word("too large to fit", "larger than memory",
                      "out of memory", "memory issues", "memory problem",
                      "without causing memory", "100gb", "100 gb",
                      "10gb", "10 gb", "chunk", "stream a file",
                      "read a large file", "large csv", "generator expression",
                      "iteratively", "line by line",
                      "public api", "requests data from",
                      "validate and clean data", "received from an external"),
            )),
            Subtopic("Functions and Modules", (
                # "Write Python code that merges overlapping sessions
                # for each user and calculates tota...", "Write a Python
                # solution to reverse a string such as \"A, B, C, D, A\"."
                _word("python solution", "using both python and sql",
                      "merges overlapping",
                      "reverse a string in python",
                      "python generator", "python generators",
                      "generators are useful in data",
                      "error handling and logging",
                      "logging in a python", "function", "decorator",
                      "generator", "iterator",
                      "lambda", "closure", "comprehension", "recursion",
                      "module", "package", "import", "args", "kwargs",
                      "scope"),
            )),
            Subtopic("Object-Oriented Programming", (
                _word("oop", "object oriented", "class", "inheritance",
                      "polymorphism", "encapsulation", "abstraction",
                      "method", "property", "dunder", "magic method",
                      "__init__", "dataclass"),
            )),
            Subtopic("Error Handling", (
                _word("exception", "error handling", "try", "except",
                      "raise", "finally", "custom error", "assertion"),
            )),
            Subtopic("Language Fundamentals", (
                # "How is memory managed in Python?", "What is Python's
                # Global Interpreter Lock, and what role does it play?",
                # "Which foundational Python concepts does the post
                # identify for learners who want to b...", "Why does the
                # post recommend learning Python, and in which
                # application areas does it ..."
                _word("memory managed in python", "global interpreter lock",
                      "python's gil", "foundational python",
                      "recommend learning python",
                      "why does the post recommend learning python",
                      "python is a", "why python",
                      "interpreted language", "dynamic typing",
                      "garbage collector", "reference counting"),
            )),
            Subtopic("Files and IO", (
                _word("useful in data pipelines",
                      "useful in a data pipeline",
                      "numeric conversion exercise",
                      "file handling", "open file", "csv module",
                      "pickle", "serialization", "serialisation",
                      "pathlib", "io", "os module"),
            )),
            Subtopic("Typing and Testing", (
                # "What is unit testing, and what does it verify?" --
                # the definitional form, which is a testing question.
                # The other two in the corpus ("Why is unit testing
                # valuable before integrating components into a larger
                # system?", "How can unit testing reduce the risk of
                # updating data pipeline code?") are about pipeline
                # change management and sit in Data Engineering, which is
                # the right answer for them.
                _word("what is unit testing", "unit testing, and what",
                      "type hint", "typing", "mypy", "annotation",
                      "unit test", "pytest", "unittest", "mock",
                      "fixture"),
            )),
            Subtopic("Pandas and Dataframes", (
                # "How would you handle missing values in a large
                # dataset using Python?"
                _word("missing values", "dataframe library",
                      "using pandas", "in pandas",
                      "pandas"),
                _word("pandas", "dataframe library", "series",
                      "groupby pandas", "apply"),
            )),
            Subtopic("General", ()),
        ),
    )


def _adf() -> Major:
    return Major(
        name="Azure Data Factory",
        blurb="Managed pipelines: activities, compute, parameters and "
              "what triggers them.",
        patterns=(
            _word("azure pipeline", "azure pipelines",
                  "azure data factory", "adf", "data factory",
                  "linked service", "integration runtime",
                  "self-hosted", "tumbling window", "event trigger",
                  "data flow", "copy activity", "lookup activity"),
        ),
        subtopics=(
            Subtopic("Pipelines", (
                # "Explain partitioning strategies and optimizations in
                # ADF.", "How can Azure Data Factory dynamically copy
                # each sheet from multiple Excel files into ...", "How
                # would you design an Azure Pipeline for 100 source files
                # in Parquet, CSV, Avro, ..."
                _word("partitioning strategies", "optimizations in adf",
                      "each sheet", "multiple excel files",
                      "100 source files", "source files in parquet",
                      "pipeline", "data pipeline", "activity",
                      "dependency", "debug", "control flow"),
            )),
            Subtopic("Activities", (
                _word("copy activity", "lookup", "foreach", "if condition",
                      "wait activity", "stored procedure activity",
                      "data flow", "notebook activity", "custom activity"),
            )),
            Subtopic("Integration Runtime", (
                _word("integration runtime", "self-hosted",
                      "self hosted", "azure ir", "compute unit"),
            )),
            Subtopic("Parameters and Variables", (
                _word("parameter", "variable", "expression",
                      "dynamic content", "trigger file"),
            )),
            Subtopic("Triggers", (
                _word("trigger", "schedule trigger", "event trigger",
                      "tumbling window", "cron"),
            )),
            Subtopic("Integration with Databricks", (
                # "How do you integrate Azure Data Factory or Airflow
                # with Databricks notebooks?", "How would you implement
                # SCD Type 2 using Azure Data Factory, with ADLS as the
                # source"
                _word("integrate azure data factory", "with databricks notebooks",
                      "scd type 2 using azure data factory",
                      "using azure data factory", "notebook",
                      "external compute", "databricks"),
            )),
            Subtopic("Incremental Loads", (
                # "How do you handle incremental loads in Azure Data
                # Factory using a watermark or timestamp?", "How would
                # you configure Azure Data Factory for incremental loads
                # from an on-premises S..."
                #
                # Its own subtopic: a watermark is a specific design a
                # candidate is asked to describe, and it is the question
                # that separates "I have used ADF" from "I can design a
                # load".
                _word("incremental load", "incremental loads",
                      "watermark", "timestamp-based load",
                      "high water mark", "last processed",
                      "delta load", "full load", "append only",
                      "batch window", "load window"),
            )),
            Subtopic("Monitoring and Error Handling", (
                # "How do you implement error handling, alerting, and
                # monitoring in ADF?", "How would you build an Azure Data
                # Factory monitoring solution using Azure Monitor", "How
                # do you handle exceptions in ADF?"
                _word("exceptions in adf", "handle exceptions",
                      "error handling", "exception handling", "alerting",
                      "monitoring solution", "azure monitor",
                      "application insights", "log analytics",
                      "retry policy", "backpressure", "failure handling",
                      "alert", "failure condition", "validate activity",
                      "data flow monitoring"),
            )),
            Subtopic("Data Flows and Transformations", (
                # "What is the difference between Pivot and Unpivot
                # transformations in Azure Data Factory?"
                _word("data flow", "data flows", "mapping data flow",
                      "transformation in azure", "pivot transformation",
                      "unpivot transformation", "derived column",
                      "aggregate", "lookup transformation",
                      "conditional split", "flatten transformation",
                      "join transformation", "expression language"),
            )),
            Subtopic("Linked Services and Datasets", (
                _word("linked service", "dataset", "data source",
                      "sink", "connection string",
                      # "How can you read and process 10 million CSV
                      # files in ADLS Gen2 using Azure Data Factory?"
                      "adls gen2", "10 million", "millions of files",
                      "self-hosted", "on-premises source",
                      "structure of a source table"),
            )),
            Subtopic("General", ()),
        ),
    )


def _data_engineering() -> Major:
    return Major(
        name="Data Engineering",
        blurb="Moving and modelling data: extraction patterns, warehouse "
              "design, and how changes are tracked.",
        patterns=(
            # The orchestration tools are here because of what happened
            # without them: "migrating a daily cron job to Airflow" had no
            # subject claim, so every subtopic competed on match length
            # and "duplicate" beat "cron". Naming the subject removes the
            # contest rather than trying to win it.
            _word("data engineering", "etl", "elt", "data warehouse",
                  "airflow", "dbt", "orchestration", "data pipeline",
                  "cron job", "scheduling",
                  "data modeling", "data modelling", "scd",
                  "slowly changing", "cdc", "change data capture",
                  "medallion architecture", "dimensional modeling",
                  "data pipeline", "data platform", "data lake",
                  "lakehouse", "olap", "oltp", "data mesh", "modern data",
                  "data quality", "data contract", "lineage",
                  "schema drift", "data ingestion", "data load",
                  "dimensional", "fact table", "dimension table",
                  "normalization", "normalisation", "batch processing",
                  "streaming", "orchestration", "data flow",
                  "reverse etl", "data catalog", "data catalogue",
                  "data profiling", "data cleansing", "data marts"),
        ),
        subtopics=(
            Subtopic("ETL and ELT", (
                _word("etl", "elt", "extract", "load", "transform",
                      "ingestion", "batch load", "full load",
                      "incremental load", "data cleaning",
                      "data cleansing", "duplicate removal",
                      "deduplication", "data validation",
                      "reverse etl"),
            )),
            Subtopic("Change Data Capture", (
                # "In a MySQL-to-PostgreSQL CDC flow using Debezium and
                # Kafka, what failure mode and verification ...?" and
                # "How would you validate row-level accuracy when moving
                # MySQL changes through a CDC pipeline?" both name CDC
                # and neither reached here while *mysql* was an SQL
                # subject pattern.
                _word("debezium", "mysql-to-postgresql", "cdc flow",
                      "cdc", "change data capture", "binlog",
                      "logical decoding", "log-based", "lsn"),
            )),
            Subtopic("Slowly Changing Dimensions", (
                _word("scd", "slowly changing", "scd type 1", "scd type 2",
                      "scd type 3", "type 2 dimension", "effective date",
                      "current flag"),
            )),
            Subtopic("Data Modeling", (
                _word("data modeling", "data modelling", "dimensional model",
                      "star schema", "snowflake schema", "fact table",
                      "dimension table", "normalization", "surrogate key",
                      "grain"),
            )),
            Subtopic("Data Warehousing", (
                _word("data warehouse", "olap", "oltp", "lakehouse",
                      "medallion", "bronze", "silver", "gold layer",
                      "data lake", "s3", "redshift", "bigquery",
                      "snowflake", "synapse", "warehouse"),
            )),
            Subtopic("Architecture and Patterns", (
                # "What is vertical scaling, and which resources does the
                # post say can be added to the data ...", "What
                # characteristics does the post associate with modern data
                # pipelines?", "What were the frequency and volume of the
                # data in your source?"
                _word("vertical scaling", "horizontal scaling",
                      "modern data pipeline", "characteristics does the post",
                      "frequency and volume", "volume of the data",
                      # "How does Data Mesh change responsibility for
                      # managing data compared with a centraliz...", "How
                      # do data marts and massively parallel processing
                      # support analytical workloads?", "What does a data
                      # pipeline orchestrate, and which implementation
                      # options does the pos...", "How can a data
                      # pipeline remain consistent and fault-tolerant"
                      "cloud big data pipeline", "big data pipeline described",
                      "engineering reliable data",
                      "reliable data pipeline",
                      "end-to-end data engineering stack",
                      "data mesh", "data marts", "data mart",
                      "massively parallel processing", "mpp",
                      "pipeline orchestrate", "orchestrates",
                      "fault-tolerant", "fault tolerance",
                      "stage 1 of the data engineering",
                      "stage 1 of the data engine",
                      "advanced data engineering skills",
                      "implementation scope", "fourth-round",
                      "data engineering foundation",
                      "modern data stack",
                      "architecture", "pattern", "design", "lambda",
                      "kappa", "micro-batch", "streaming", "batch",
                      "event-driven",
                      # "How do I handle schema evolution in my data
                      # pipelines?", "what happens when a column changes
                      # from int to"
                      "schema evolution", "column changes from",
                      "type widening", "additive change",
                      "breaking change", "backward compatible schema",
                      # "According to the post, what should CI/CD for
                      # data pipelines automate?", "A team wants processed
                      # files removed from the source location but
                      # preserved in an archive ...", "An application team
                      # wants automated database maintenance and Multi-AZ
                      # availability."
                      "ci/cd", "ci cd", "continuous delivery for data",
                      "data pipeline automation", "automate deployment",
                      "archive", "archiving", "file retention",
                      "high availability", "multi-az", "failover",
                      "disaster recovery", "storage tier", "storage cost"),
            )),
            Subtopic("Orchestration and Scheduling", (
                _word("orchestration", "scheduler", "airflow", "dag",
                      "dependency", "upstream", "downstream", "cron",
                      "backfill"),
            )),
            Subtopic("Data Quality", (
                # "What steps did you take to handle dirty or missing
                # data?", "Which techniques does the post associate with
                # cleaning and formatting database records?", "Deduplicate
                # records by user_id while keeping the row with the latest
                # timestamp.", "Detect duplicate records in a
                # transactional table and delete extras safely.", "Why
                # does the provided sample output appear inconsistent
                # with the stated metric?"
                _word("dirty or missing", "dirty data", "missing data",
                      "inconsistent with the stated",
                      "sample output appear", "appear inconsistent",
                      "consistency check",
                      "cleaning and formatting",
                      "formatting database records",
                      "data profiling", "data quality", "validation",
                      "dq", "expectation",
                      "lineage", "schema drift", "contract",
                      "data cleaning", "data cleansing",
                      "null handling", "outlier", "deduplication",
                      "deduplicate", "deduplicating", "duplicate record",
                      "duplicates", "survivorship", "latest timestamp",
                      "referential integrity", "reconciliation"),
            )),
            Subtopic("File Formats and Encoding", (
                # JSON, XML, CSV, Parquet, Avro, ORC as a subject in their
                # own right rather than as loose labels. 16 labels, but
                # they recur across every platform and a candidate is
                # asked about them as format choice, not as trivia.
                _word("json", "xml", "parquet", "avro", "orc file",
                      "csv file", "protobuf", "thrift", "pickle",
                      "serialization", "schema registry", "compression",
                      "flat file", "delimited file", "character encoding",
                      "nested json", "flattening nested",
                      "semi-structured", "unstructured data"),
            )),
            Subtopic("Project and Experience", (
                # "Describe an end-to-end data pipeline you built and took
                # to production.", "Explain your experience with building
                # scalable data pipelines.", "Describe a challenging Data
                # Engineering problem you faced and how you resolved it.",
                # "How many records have you processed in your data
                # engineering work?", "How much data did you handle each
                # day, and what was the business case for the project?"
                #
                # A real revision area with a real shape -- what to say,
                # in what order -- and 40-odd of these questions ask for
                # it. It was being filed under Architecture and Patterns,
                # which is about the design, not the account of it.
                _word("describe a data engineering project",
                      "describe a project", "data engineering project",
                      "end-to-end data pipeline you built",
                      "to production", "challenging data engineering",
                      "scalable data pipeline", "your experience building",
                      "tell me about your project", "walk me through a project",
                      "a project you", "explain your experience",
                      "your experience with", "have you built",
                      "what did you build", "describe your experience",
                      "describe a data pipeline", "pipeline you have built",
                      "records have you processed",
                      "data engineering work", "how much data did you handle",
                      "business case for the project",
                      "explain an end-to-end data pipeline",
                      "end-to-end data engineering flow",
                      "data engineering pipeline you",
                      "during an interview",
                      # "How should a generic data-pipeline resume bullet
                      # be revised to communicate impact?", "What
                      # benefits does the post claim for data teams
                      # using this template?", "Where would you place a
                      # project's main usage instructions, and what
                      # information shou..."
                      "promote data engineering code",
                      "higher environment",
                      "resume bullet", "data teams using this template",
                      "preparation advice", "main usage instructions",
                      "birlasoft", "production-oriented scenarios",
                      # "How does the post describe SQL's role across the
                      # stages of a data engineering pipeline?", "Why does
                      # the author argue that data engineers still need SQL
                      # when using a modern dat...", "Which components
                      # support the model lifecycle in this project, and
                      # what role does eac..."
                      "sql's role", "across the stages",
                      "data engineers still need", "still need sql",
                      "model lifecycle", "components support the",
                      "securing data pipelines",
                      # "How would you walk through a previous data
                      # science project from its beginning to i...", "How
                      # would you implement error handling and logging in
                      # a Python-based data pipeline", "How do you
                      # containerize and deploy data pipelines?"
                      "walk through a previous",
                      "from its beginning to",
                      "error handling and logging",
                      "containerize and deploy data pipelines",
                      "containerise and deploy data pipelines"),
            )),
            Subtopic("Governance, Compliance and Audit", (
                # "How do you handle PII and GDPR compliance in data
                # pipelines?", "How would you manage audit logging for
                # data pipeline activities?", "How can the process
                # distinguish an existing customer whose relevant fields
                # changed fro...", "Why is a simultaneous login from
                # multiple IP addresses considered a data integrity
                # concern?", "How would you validate row-level accuracy
                # when moving MySQL changes through a CDC ...", "How do
                # you reprocess data after a job fails?", "How would you
                # handle a failing ADF data flow caused by a schema
                # mismatch"
                _word("row-level accuracy", "validate row-level",
                      "reprocess data", "reprocess",
                      "failing adf data flow", "schema mismatch",
                      "data integrity", "integrity concern", "pii", "gdpr",
                      "compliance", "audit logging",
                      "audit log", "audit trail", "data governance",
                      "retention policy", "right to be forgotten",
                      "existing customer whose", "relevant fields changed",
                      "master data", "golden record", "authoritative source",
                      "regulatory", "data catalog and lineage"),
            )),
            Subtopic("Pipelines and Troubleshooting", (
                # "How do you reprocess data after a job fails?", "A data
                # pipeline fails in production but works in development.
                # How would you approach ...", "How do you manage and
                # monitor pipeline failures in Azure Data Factory?", "What
                # happens when data pipeline execution fails?", "What
                # objectives should guide data-pipeline optimization",
                # "What performance optimizations have you implemented in
                # data pipelines or Spark jobs", "Why is unit testing
                # useful before components are integrated into a larger
                # data pipel...", "What model and data conditions does the
                # post say should be monitored?"
                _word("unit testing", "components are integrated",
                      "testing before", "should be monitored",
                      "conditions should be monitored",
                      "after a job fails", "job fails",
                      "pipeline fails",
                      "pipeline failure", "fails in production",
                      "works in development", "root cause", "troubleshoot",
                      "debugging", "incident", "breakdown", "production issue",
                      "monitor pipeline", "pipeline reliability",
                      "why did it fail", "resolve a failure",
                      # "Why would choosing approximately 400 cores for
                      # this file be considered excessive?"
                      "cores for this file", "400 cores",
                      "excessive", "how many cores",
                      "pipeline execution fails", "execution fails",
                      "objectives should guide", "pipeline optimization",
                      "optimizations have you implemented",
                      "optimizations have you applied"),
            )),
            Subtopic("General", ()),
        ),
    )


def _aws() -> Major:
    return Major(
        name="AWS and Cloud",
        blurb="The managed services a data platform is usually built on.",
        patterns=(
            # "rds" was missing entirely: "What database maintenance
            # capabilities does the post attribute to Amazon RDS?" and
            # "Which security controls does the post identify for
            # Amazon RDS?" both fell through to the catch-all.
            _word("aws", "amazon web services", "amazon rds", "rds",
                  "s3", "redshift", "glue",
                  "athena", "emr", "lambda", "dynamodb", "cloudwatch",
                  "kinesis", "sqs", "sns", "eventbridge", "vpc", "iam",
                  "s3 bucket", "object storage", "bigquery", "snowflake",
                  "synapse", "cloud storage", "serverless"),
        ),
        subtopics=(
            Subtopic("Storage", (
                _word("s3", "bucket", "object storage", "storage class",
                      "lifecycle rule"),
            )),
            Subtopic("RDS and Managed Databases", (
                # "How does the post describe the role of Multi-AZ
                # deployments in Amazon RDS?", "How does the post say
                # Amazon RDS supports database performance monitoring?",
                # "What database maintenance capabilities does the post
                # attribute to Amazon RDS?", "What scalability
                # capabilities does the post claim Amazon RDS provides?",
                # "Which security controls does the post identify for
                # Amazon RDS?"
                _word("amazon rds", "rds", "multi-az deployment",
                      "database maintenance", "maintenance capabilities",
                      "scalability capabilities",
                      "database performance monitoring"),
            )),
            Subtopic("Glue and ETL", (
                # "Which technologies does the post explicitly associate
                # with the event-based AWS ETL pipel..."
                _word("event-based aws etl", "event-based etl",
                      "glue", "crawler", "catalog service", "glue job"),
            )),
            Subtopic("Compute", (
                _word("emr", "ec2", "lambda", "ecs", "eks",
                      "serverless", "compute"),
            )),
            Subtopic("Data Warehouses", (
                _word("redshift", "athena", "bigquery", "snowflake",
                      "synapse"),
            )),
            Subtopic("BigQuery", (
                # "How do BYROW, BYCOL, and SCAN differ in how they
                # process data?", "What are MAGIC TABLES in SQL?".
                # BigQuery-specific vocabulary with no home until now, so
                # it fell through to the broadest subject.
                _word("bigquery", "byrow", "bycol", "magic table",
                      "dry run", "dml table", "permanent table",
                      "partition decorator", "cluster column"),
            )),
            Subtopic("NoSQL and Messaging", (
                _word("dynamodb", "kinesis", "sqs", "sns", "eventbridge",
                      "nosql", "key-value", "kafka", "messaging",
                      "message queue", "pub/sub", "topic",
                      "stream processing", "hive", "hadoop",
                      "relational database"),
            )),
            Subtopic("Cloud Storage and Identity", (
                _word("adls", "azure blob", "key vault", "keyvault",
                      "secret", "managed identity", "service principal",
                      "credential", "adf", "data factory"),
            )),
            Subtopic("Networking and IAM", (
                # "role" reaches "the role of the team" and "a remote
                # role"; the compound is the IAM term.
                _word("vpc", "iam", "iam role", "service role",
                      "execution role", "role assignment",
                      "role-based", "policy", "security group",
                      "subnet", "cross-account",
                      # "What is an AWS Target Group, and how is it
                      # used?", "How are changes applied in AWS Cloud?",
                      # "How can AWS infrastructure be created as a
                      # scalable setup?"
                      "target group", "load balancer", "elb", "alb",
                      "health check", "scaling setup", "scalable setup",
                      "infrastructure be created", "provisioning",
                      "public subnet", "private subnet",
                      "nat gateway", "route table"),
            )),
            Subtopic("Monitoring and Operations", (
                # "How do you monitor an application and configure
                # alerting?", "AI observability"
                _word("monitor an application", "monitoring and alerting",
                      "configure alerting", "observability", "cloudwatch",
                      "alert rule", "cloudwatch metric", "custom metric",
                      "metric filter", "metric alarm", "dashboarding",
                      "service health", "status page"),
            )),
            Subtopic("Multi-Cloud Comparison", (
                # "How do the post's AWS, Azure, and GCP service mappings
                # differ across the storage, warehousing, and ... layers",
                # "Which AWS or GCP services have you used, and why were
                # they appropriate?"
                _word("gcp", "google cloud", "multi-cloud",
                      "service mapping", "service mappings",
                      "aws, azure, and gcp", "across aws",
                      "equivalent service", "which cloud",
                      "aws or gcp", "azure or aws"),
            )),
            Subtopic("General", ()),
        ),
    )


def _interview() -> Major:
    return Major(
        name="Interview Preparation",
        blurb="The non-technical half: how to answer, and what to say when "
              "you have not used something.",
        patterns=(
            # No "how would you" and no "walk me through". Both are
            # interrogative forms, not subject matter: at subject level
            # they named Interview Preparation for any ADF, Power BI or
            # pipeline-design question that opened with them, and at
            # subtopic level they did the same. What is left names the
            # subject.
            _word("interview", "behavioral", "behavioural", "scenario",
                  "system design", "mock interview", "interview preparation",
                  "aptitude", "troubleshoot", "debugging", "root cause",
                  "incident", "production issue",
                  "strength", "weakness", "feedback",
                  # Compounds, and they are here for the reason the
                  # docstring on classify() gives for "Spark SQL": a
                  # question can name a subject *and* the context it is
                  # asked in. "What should a data engineering candidate
                  # prioritize when preparing for an interview?" contains
                  # "data engineering", so pass 0 named Data Engineering
                  # (17 characters) and this subject never saw it -- even
                  # though the question is about interview technique.
                  # A longer, more specific claim wins.
                  "data engineering interview",
                  "data engineering candidate",
                  "preparing for an interview",
                  "prepare for an interview",
                  "preparation for an interview",
                  "interview challenge",
                  "interview problem",
                  "interview question",
                  "interview red flag",
                  "coding interview",
                  "live coding",
                  "coding round",
                  "pyspark roadmap", "pyspark preparation",
                  "pyspark syllabus", "preparation syllabus",
                  "preparation outline", "pyspark stages of the roadmap",
                  "first-round preparation",
                  "first interview",
                  "four-week roadmap",
                  "four week roadmap",
                  "candidate revise",
                  "candidate prepare",
                  "should a candidate",
                  "candidate consider"),
        ),
        subtopics=(
            Subtopic("Troubleshooting", (
                # "Which performance problems are presented, and what
                # does the post say interviewers wa..."
                _word("performance problems are presented",
                      "troubleshoot", "debug", "root cause",
                      "incident", "production issue", "failure",
                      "diagnos", "breakdown"),
            )),
            Subtopic("System Design", (
                _word("system design", "scalability", "high availability",
                      "load balancing", "distributed system",
                      "capacity planning", "bottleneck"),
            )),
            Subtopic("Architecture", (
                _word("architecture", "design pattern", "data flow",
                      "reference architecture"),
            )),
            Subtopic("Scenario Questions", (
                # "how would you" and "walk me through" were here and
                # both were interrogative forms rather than subject
                # matter. They matched twelve characters of almost any
                # question and pulled ADF, Power BI and pipeline design
                # questions into Interview Preparation -- a technical
                # question filed under interview technique, which is the
                # opposite of what this subtopic is for.
                #
                # What remains are the words that actually denote an
                # open-ended scenario. A genuine "how would you handle a
                # data quality incident" still reaches here by inheriting
                # its post's placement, which is the honest result when
                # the question names no technology.
                _word("scenario", "scenario question", "case study",
                      "real-world situation", "how would you handle",
                      "how would you deal", "how would you approach a",
                      "how would you respond", "how would you investigate"),
            )),
            Subtopic("Behavioral", (
                # "A Data Engineering candidate has strong technical
                # skills but is receiving few interv..."
                _word("receiving few interview", "strong technical skills",
                      "behavioral", "behavioural", "conflict", "feedback",
                      "strength", "weakness", "teamwork",
                      "tell me about yourself",
                      # "How do I handle disagreements between data
                      # engineering and business stakeholders?", "How do I
                      # manage communication when multiple teams depend on
                      # your data pipelines?", "How do I handle tight
                      # deadlines or changing priorities during a sprint?",
                      # "Have you mentored junior engineers, and how do
                      # you perform code reviews?"
                      "disagreement", "disagreements", "handling conflict",
                      "changing priorities", "tight deadline",
                      "deadline", "multiple teams depend",
                      "managed junior", "mentored junior",
                      "mentoring junior", "code review", "code reviews",
                      "structure and communicate an insight",
                      # "A dashboard shows that sales have dropped. How
                      # should a data analyst communicate this ...?"
                      "communicate a finding", "communicate this",
                      "present a finding", "tell a stakeholder",
                      "non-technical stakeholder", "data storytelling",
                      "business stakeholder"),
            )),
            Subtopic("Practice Platforms and Preparation", (
                # "How do LeetCode, HackerRank, and StrataScratch differ
                # in their approach to SQL interview questions?", "How
                # does the post recommend selecting problems for a DSA
                # preparation plan?", "What binary-search variations
                # should an interview candidate prepare to recognize?"
                # "Where does the post recommend practicing SQL, and what kinds of "
                # "problems should candidates ...?", "Which SQL capabilities "
                # "are prioritized in Week 1 of the roadmap?", "Which SQL "
                # "capabilities are tested by this challenge, and why?"
                _word("recommend practicing sql", "practicing sql",
                      "test sql queries quickly", "prioritized in week 1",
                      "tested by this challenge", "which sql capabilities",
                      "which python capabilities", "which pytorch capabilities",
                      "technical progression", "roadmap stage",
                      "weekly plan", "four weeks",
                      # "What preparation approach does the post
                      # recommend for a data engineering interview?",
                      # "What should a data engineering candidate
                      # prioritize when preparing for an interview?",
                      # "Which areas of preparation does the post
                      # recommend for experienced data engineering
                      # interviewees?", "Why does the post recommend paper
                      # and whiteboard practice?", "Which medium-level
                      # Round 3 problems were asked", "which sql skills does
                      # this post identify as important"
                      "in an interview", "for an interview",
                      "in an sql interview", "interview challenge",
                      "coding task described", "live coding",
                      "ready to apply", "should a candidate",
                      "rapid recall", "interview red flag",
                      "red flag", "beyond the happy path",
                      "happy path", "should a sql interview",
                      "amazon sde", "sde-1", "reverse-engineer",
                      "memorizing pyspark", "memorizing",
                      "amazon interview", "myntra", "deloitte",
                      "assessed in the", "were assessed",
                      "breadth of technical knowledge",
                      "final interview round",
                      "coding tasks are highlighted",
                      "recall and apply", "preparation priorities",
                      "pyspark roadmap", "pyspark preparation",
                      "pyspark syllabus", "preparation syllabus",
                      "preparation outline", "weeks 2 through",
                      "broad areas organize",
                      "preparation process",
                      "learning strategy when preparing",
                      "everyday sql work",
                      "interview-question collection",
                      "listed interview topics",
                      "interview topics",
                      "ranking, indexing, and query-optimization",
                      "advanced sql topics",
                      "preparing for the dsa",
                      "interview questions using",
                      "preparing for an interview",
                      "prepare for an interview",
                      "preparation approach", "areas of preparation",
                      "prepare when", "should strengthen",
                      "prepare for interviews",
                      "paper and whiteboard", "round 3 problem",
                      "round 1", "round one", "round 2",
                      "interview scenarios test",
                      "which sql skills does",
                      "leetcode", "hackerank", "stratascratch",
                      "practice platform", "dsa preparation",
                      "preparation plan", "study plan",
                      "sql interview questions", "practice sql",
                      "practicing sql", "practise sql",
                      "binary-search variation", "binary search variation",
                      "mock interview", "aptitude",
                      "hands-on practice", "memorization",
                      "memorisation", "consistent practice"),
            )),
            Subtopic("Interview Process and Preparation", (
                # "How was the Data Engineer interview process described
                # in the post structured?", "How does the post recommend
                # efficiently preparing for company-specific c..."
                _word("interview process", "first round", "second round",
                      "company-specific", "round 1", "round one",
                      "hr round", "managerial round", "technical round",
                      "interview-preparation", "interview preparation",
                      "roadmap", "week 1", "week one", "checklist",
                      "preparation roadmap", "what to prepare"),
            )),
            Subtopic("General", ()),
        ),
    )


def _machine_learning() -> Major:
    return Major(
        name="Machine Learning and AI",
        blurb="Models and the pipeline around them: features, training, "
              "evaluation and deployment.",
        patterns=(
            # Deliberately no bare "ai" or "ml". Both were here and both
            # were too greedy: a question about which storage type suits
            # exploratory work "and AI/ML" was being filed as a machine
            # learning question because the words appeared in it. The
            # subject names itself.
            # "unsupervised" was a subtopic keyword only, so "Which
            # clustering and dimensionality-reduction methods are listed
            # for unsupervised learning?" matched no subject and fell to
            # the catch-all.
            _word("unsupervised learning", "supervised learning",
                  "machine learning", "deep learning", "neural network",
                  "ai engineer", "ai engineer", "artificial intelligence",
                  "llm", "large language model", "generative ai",
                  "generative model", "mlflow", "computer vision",
                  "nlp", "natural language processing", "ml model",
                  "ml models", "ml pipeline", "mlops", "transformer",
                  "reinforcement learning"),
        ),
        subtopics=(
            Subtopic("Deep Learning and NLP", (
                # "How do machine learning, deep learning, and neural
                # networks relate ...", "Which deep learning, NLP,
                # transformer, and reinforcement learning topics appear
                # in the roadmap?"
                _word("deep learning", "neural network", "neural networks",
                      "nlp", "natural language processing", "transformer",
                      "transformers", "attention mechanism", "cnn", "rnn",
                      "lstm", "embedding model", "word embedding",
                      "sequence model"),
            )),
            Subtopic("ML Roadmaps and Curriculum", (
                # Removed: "ML Projects and Experience". A sweep for
                # project wording returned two questions -- "Tell me
                # about your project and explain it end to end." and
                # "Describe your project experience, including specific
                # examples and details about your role an..." -- and
                # neither names machine learning. They are account-of-a-
                # project questions and belong in Data Engineering /
                # Project and Experience, where both now sit. A
                # subtopic scoped to ML that only ever held non-ML
                # questions was a category with a name on it and nothing
                # in it.
                #
                # "How did you leverage machine learning algorithms in
                # your project?" and "How did you aggregate and manipulate
                # tables in SQL, build a machine-learning model ...?" do
                # name it, and both reach Supervised Learning or Model
                # Deployment in Practice instead.
                #
                # "Which foundational machine learning concepts and data
                # activities are included in the ...", "What
                # prerequisites does the roadmap identify for beginning
                # machine learning?", "What deep learning topics does the
                # roadmap mark as optional".
                #
                # A distinct shape of question: what to learn, in what
                # order, rather than how any of it works.
                _word("ml roadmap", "learning roadmap", "roadmap for",
                      "prerequisites", "beginner path",
                      "learning path", "curriculum", "what to learn",
                      "topics are included", "topics does the roadmap",
                      "which topics", "study order",
                      "subjects are included",
                      "concepts and data activities",
                      "progression from", "progression does",
                      "advanced topics", "marked as optional",
                      "topics are optional", "optional for advanced"),
            )),
            Subtopic("Supervised Learning", (
                # Bare "label" and "feature" were here. Both are
                # everyday words -- "unrelated label", "a feature flag"
                # -- and both reached this subtopic from questions that
                # have nothing to do with machine learning. What
                # supervised learning is actually named by is a
                # labelled example.
                _word("supervised", "regression model", "classification "
                      "model", "predictive model", "labeled data",
                      "labelled data", "labeled dataset", "labelled dataset",
                      "target variable", "ground truth",
                      "training data", "train and test", "decision tree",
                      "random forest", "gradient boosting", "xgboost",
                      "linear regression", "logistic regression",
                      "similarity", "prediction model"),
            )),
            Subtopic("Unsupervised Learning", (
                # "Which clustering and dimensionality-reduction methods
                # are listed for unsupervised le..."
                _word("clustering and dimensionality",
                      "dimensionality-reduction methods",
                      "unsupervised", "clustering", "k-means", "kmeans",
                      "dimensionality reduction", "pca", "anomaly",
                      "recommendation", "collaborative filtering"),
            )),
            Subtopic("Feature Engineering and Pipelines", (
                # "feature engineering" and "feature store", not
                # "feature".
                _word("feature engineering", "feature store",
                      "feature transformation", "feature importance",
                      "data preparation for ml", "model pipeline",
                      "training pipeline", "ml pipeline", "labeling"),
            )),
            Subtopic("Fine-Tuning and Alignment", (
                # "When does the post recommend using RLHF rather than
                # DPO or GRPO?", "Which two parallelism approaches does
                # the post name under distributed training?"
                _word("rlhf", "dpo", "grpo", "fine-tuning", "finetuning",
                      "fine tune", "distributed training",
                      "parallelism approaches", "alignment",
                      "human feedback", "preference optimization"),
            )),
            Subtopic("Model Deployment in Practice", (
                # "Describe the end-to-end flow of the real-time air
                # quality data and machine learning ...", "How do machine
                # learning and real-time processing differ within the
                # stack described b...", "Which components support the
                # model lifecycle in this project, and what role does
                # each ..."
                _word("end-to-end flow", "real-time processing",
                      "model lifecycle", "ai and data system",
                      "stack described", "in this project"),
            )),
            Subtopic("Model Evaluation and Deployment", (
                # "How are model serving, experiment tracking, and
                # retraining organized across the syst...", "What is the
                # generation-verification gap, and why can separating
                # generation from eval..."
                _word("model serving", "experiment tracking",
                      "retraining", "generation-verification",
                      "model evaluation", "model deployment", "deploy a model",
                      "model monitoring", "drift", "accuracy",
                      "precision and recall", "f1 score", "confusion matrix",
                      "cross validation", "hyperparameter", "mlops",
                      "inference", "a/b test", "model registry",
                      "four phases", "phases of the"),
            )),
            Subtopic("Classification and Ranking", (
                # "How does the shown ai_classify query prioritize a
                # platform support ticket?", and the sentiment and
                # categorisation questions that sit alongside it.
                _word("ai_classify", "classify query", "support ticket",
                      "sentiment analysis", "classify a ticket",
                      "text classification", "intent detection",
                      "categorize", "categorise"),
            )),
            Subtopic("Generative AI and Agents", (
                # "agents" and "agent" also reach "user-agent", which
                # is how a string header was classified as generative
                # AI. The compound is the term.
                _word("ai agent", "ai agents", "autonomous agent",
                      "agent framework", "multi-agent", "multimodal",
                      "prompt engineering", "rag", "retrieval augmented",
                      "vector database", "embedding", "embeddings",
                      "chatgpt", "copilot", "foundation model",
                      # "How do generative models, large language models,
                      # and transformers relate ...", "What concerns must
                      # be addressed when moving an AI application beyond
                      # a prototype?", "Which generative AI capabilities
                      # should a Data Engineer learn".
                      "generative ai", "generative model",
                      "large language model", "llm application",
                      "beyond a prototype", "ai application",
                      "support ai systems", "ai system architecture",
                      "ai and data system"),
            )),
            Subtopic("General", ()),
        ),
    )


def _algorithms() -> Major:
    return Major(
        name="Data Structures and Algorithms",
        blurb="The coding half of an interview: choosing an approach and "
              "proving it is right.",
        patterns=(
            # Bare "algorithm" is not a subject name: it appears in
            # "hash join algorithm", "sort algorithm", "greedy algorithm"
            # and a dozen other questions that belong to the platform
            # they are asked on. The subject matches phrases, plus the
            # specific families below, which is what identifies it.
            _word("algorithm problem", "algorithm technique",
                  "algorithm technique", "algorithm pattern",
                  "algorithm family", "dsa", "data structure and algorithm",
                  "data structures and algorithm", "leetcode",
                  "coding problem", "problem-solving process",
                  "complexity analysis", "time complexity",
                  "space complexity", "big-o"),
        ),
        subtopics=(
            Subtopic("Greedy and Algorithm Families", (
                # "What reasoning characterizes a greedy algorithm, and
                # where is it commonly applied?", "Which algorithm
                # techniques were identified for Maximum Profit in Job
                # Scheduling?"
                _word("greedy", "maximum profit in job scheduling",
                      "divide and conquer", "backtracking",
                      "algorithm families", "algorithm techniques",
                      "characterizes a greedy", "common algorithm",
                      "pattern recognition", "recognize the problem"),
            )),
            Subtopic("Streams and Counting Structures", (
                # "Given a stream of integers, design a data structure
                # that can return the median ...", "Which structure or
                # algorithm can approximate the number of unique
                # elements in a multiset?"
                _word("stream of integers", "running median",
                      "median of a stream", "unique elements in a multi",
                      "count-min sketch", "bloom filter", "lru cache",
                      "min stack", "frequency map", "frequency table",
                      "approximate the number of unique"),
            )),
            Subtopic("Practice and Progression", (
                # "How should a candidate progress after mastering the
                # core DSA problems?", "What should you do after
                # repeatedly struggling with a DSA problem?", "What
                # review schedule does the post recommend for solving
                # DSA problems repeatedly?", "Which techniques does the
                # post associate with the two Round 2 DSA problems?"
                _word("core dsa", "dsa problem", "dsa preparation",
                      "review schedule", "solving dsa repeatedly",
                      "struggling with a dsa", "round 2 dsa",
                      "mastering the core", "repeatedly struggling",
                      "solve dsa", "practice dsa", "dsa round",
                      "algorithm patterns were covered",
                      "problem-solving process"),
            )),
            Subtopic("Searching and Sorting", (
                _word("binary search", "linear search", "sorting",
                      "sort algorithm", "quick sort", "merge sort",
                      "lower bound", "upper bound", "rotated sorted",
                      "two sum", "sorted array"),
            )),
            Subtopic("Arrays and Strings", (
                # "Which two exercises focus on analyzing letters in a
                # word?"
                _word("analyzing letters in a word", "letters in a word",
                      "array", "string manipulation", "reverse a string",
                      "substring", "anagram", "palindrome", "deduplicate",
                      "merge overlapping", "flatten a nested list",
                      "character frequency", "run-length"),
            )),
            Subtopic("Two Pointers and Sliding Window", (
                _word("two pointer", "sliding window", "sliding-window",
                      "window function pattern", "longest substring",
                      "contiguous subarray"),
            )),
            Subtopic("Specialised Structures", (
                # "What operations does a SkipList support, and why is
                # its layered structure useful?", "Given person(PersonID,
                # Name, Score) and friend(pid, fid), what do these tables
                # represent ...?"
                _word("skiplist", "skip list", "b-tree", "b tree",
                      "red-black", "hash table", "trie structure",
                      "friend(pid", "personid, name, score",
                      "layered structure", "concurrent data structure"),
            )),
            Subtopic("Linked Lists", (
                _word("linked list", "linked-list", "node class",
                      "reverse a linked list", "cycle detection"),
            )),
            # Removed: "Stacks and Queues". A regex sweep of every
            # question in the corpus for stack, queue, deque, BFS or DFS
            # returned five hits and all five were incidental -- "the
            # legacy notification queue", "What technology stack do you
            # prefer", "governance, orchestration, and monitoring". There
            # is no stack or queue content here, and a revision area with
            # nothing in it is an arbitrary category rather than a useful
            # one. Trees and Graphs keeps its own "depth-first" and
            # "breadth-first" patterns, which is where a traversal
            # question would go if the corpus ever contained one.
            Subtopic("Trees and Graphs", (
                # "Detect a cycle in a directed graph representing trip
                # routes between cities."
                _word("directed graph", "cycle in a graph",
                      "cycle detection in a", "binary tree",
                      "tree traversal", "depth-first",
                      "breadth-first", "dfs", "bfs", "graph traversal",
                      "shortest path", "tree hierarchy", "adjacency",
                      "topological sort", "trie"),
            )),
            Subtopic("Dynamic Programming", (
                _word("dynamic programming", "dynamic-programming",
                      "memoization", "tabulation", "fibonacci",
                      "knapsack", "coin change", "longest common subsequence",
                      "longest increasing subsequence"),
            )),
            Subtopic("Puzzles Over Data", (
                # The largest group in this subject by far: "Find
                # customers who bought products every month in a year",
                # "Group users by login streaks of three or more
                # consecutive days", "Given the rides table, find the
                # most frequently traveled route in the past 30 days".
                # Plus the imperative forms that arrived after the first
                # pass: "How would you identify customers who purchased
                # in every category?", "How would you return the top five
                # highest-paid employees for each department?", "How
                # would you find the employee with the fifth-highest
                # salary in each department?".
                #
                # These read as SQL questions but are asked as puzzles,
                # and a candidate revises them as pattern recognition
                # rather than as syntax. Several also appear as Python
                # requests, so they are asked whichever way round.
                _word("find customers who", "find users who",
                      "find products that", "find departments where",
                      "find employees", "find the most frequently",
                      "login streak", "consecutive days",
                      "consecutive months", "every month in a year",
                      "zero sales", "denormali", "self-join",
                      "given the rides", "given a rides",
                      "given ids", "given the one-column",
                      "separate the numeric", "travelled route",
                      "traveled route", "sales for three",
                      # "How would you fetch the top 3 categories
                      # contributing to 80% of total revenue?", "Using
                      # orders, order_items, and products, find the three
                      # product categories with the ...", "How would you
                      # identify users who upgraded to premium but never
                      # used the features?", "Why is certificate 7 marked
                      # inactive for candidate b under the stated
                      # end-exclusive rule"
                      "contributing to 80%", "top 3 categories contributing",
                      "order_items", "three product categories",
                      "upgraded to premium", "end-exclusive",
                      "marked inactive", "identified customers who",
                      "identify customers who", "identify customers whose",
                      "return the top five", "return the top",
                      "for each department", "in each department",
                      "highest-paid employees",
                      "spent the most", "who spent",
                      "purchased in every", "highest sale for each",
                      "every category", "delete every nth",
                      "nth row", "above their managers",
                      "more than their managers", "earn more than",
                      "exactly one order", "one order in",
                      "afford under", "affordability condition",
                      "20-6-20", "on_time_flag", "session time from",
                      "total revenue per customer", "top five customers",
                      "reproduce the project's result",
                      "most frequently traveled"),
            )),
            Subtopic("General", ()),
        ),
    )


def _devops() -> Major:
    return Major(
        name="Git and DevOps",
        blurb="Version control, infrastructure as code, and how a change "
              "reaches production.",
        patterns=(
            _word("git", "devops", "dataops", "ci/cd", "ci cd",
                  "infrastructure as code", "terraform", "kubernetes",
                  "docker", "containerization", "version control"),
        ),
        subtopics=(
            Subtopic("Version Control with Git", (
                # "branch" reaches "a branch of study".
                _word("git", "git branch", "branch strategy",
                      "feature branch", "branch protection",
                      "branching model",
                      "merge conflict", "pull request",
                      "rebase", "cherry-pick", "version control",
                      ".gitignore", "commit history", "tag a release"),
            )),
            Subtopic("CI/CD for Data", (
                # "How do I containerize and deploy data pipelines?",
                # "How do I implement versioned data pipelines in
                # Databricks notebooks using Git integration ...?", "How
                # do I promote code from one environment to another?",
                # "How do the src/ and tests/ directories support project
                # quality?"
                _word("ci/cd", "ci cd", "continuous integration",
                      "continuous delivery", "continuous deployment",
                      "build pipeline", "automate deployment",
                      "pipeline automation", "unit test in pipeline",
                      "deployment pipeline", "release pipeline",
                      "containerize and deploy",
                      "containerise and deploy", "git integration",
                      "versioned data pipeline", "versioning pipeline",
                      "promote code from one environment",
                      "promote code between environment",
                      "src/ and tests/", "directories support project",
                      "project quality",
                      # "What information should a README.md contain
                      # according to the proposed structure?", "What
                      # purposes do the config/, data/, and docs/
                      # directories serve ...?"
                      "readme", "readme.md", "config/, data/",
                      "proposed structure", "project structure",
                      "directory structure", "repository structure"),
            )),
            Subtopic("Infrastructure as Code", (
                _word("terraform", "infrastructure as code", "iac",
                      "cloudformation", "pulumi", "bicep",
                      "provisioning", "reusable module", "state file"),
            )),
            Subtopic("Containers and Orchestration", (
                _word("kubernetes", "docker", "containerization",
                      "container image", "helm", "kubernetes pod",
                      "pod spec", "deployment yaml",
                      "dockerfile", "image registry"),
            )),
            Subtopic("Observability", (
                _word("observability", "monitoring and alerting",
                      "application monitoring", "alerting", "logging",
                      "metrics", "tracing", "data quality monitoring",
                      "pipeline observability"),
            )),
            Subtopic("General", ()),
        ),
    )


def _bi() -> Major:
    return Major(
        name="Business Intelligence and Reporting",
        blurb="Turning the data into something a non-technical person acts "
              "on.",
        patterns=(
            _word("power bi", "tableau", "looker", "quicksight",
                  "business intelligence", "bi tool", "reporting tool"),
        ),
        subtopics=(
            Subtopic("Reporting and Dashboards", (
                _word("reporting", "report", "dashboard", "visualization",
                      "dashboard design", "kpi", "scorecard",
                      "self-service", "semantic layer"),
            )),
            Subtopic("DAX and Measures", (
                _word("dax", "measure", "calculated column",
                      "running total dax", "time intelligence"),
            )),
            Subtopic("Spreadsheets as a Data Tool", (
                # A genuinely asked-about topic in this corpus: "Which
                # Excel capabilities does the post identify as
                # foundational ...", "pivot tables, VLOOKUP, XLOOKUP,
                # conditional formatting, and Power Query".
                _word("excel", "spreadsheet", "vlookup", "xlookup",
                      "pivot table", "conditional formatting",
                      "power query", "power pivot", "vba", "macro",
                      "pivot report"),
            )),
            Subtopic("Data Modeling in BI", (
                # "How should many-to-many relationships be handled in a
                # Power BI data model?"
                _word("many-to-many", "data model", "star schema in power bi",
                      "relationship in power bi", "dimension and measure",
                      "row-level security in power bi", "slowly changing in bi"),
            )),
            Subtopic("Fabric and Synapse", (
                # "How does Microsoft Fabric differ from Azure Synapse
                # Analytics and Power BI?"
                _word("microsoft fabric", "fabric", "azure synapse",
                      "synapse analytics", "power bi relationship",
                      "power bi with sql", "sql supporting business intelligence"),
            )),
            Subtopic("Communicating with Data", (
                _word("communicate a finding", "data storytelling",
                      "stakeholder", "business insight", "explain the data",
                      "actionable insight", "communicate this"),
            )),
            Subtopic("General", ()),
        ),
    )


def _azure_platform() -> Major:
    return Major(
        name="Azure Platform",
        blurb="The managed services around a Microsoft data platform, "
              "separate from the orchestration tool.",
        patterns=(
            # The compound service names are here because a specific
            # technology must beat a generic concept. "What is the role
            # of Azure Key Vault in securing data pipelines?" was placed
            # by "data pipeline" (14 characters, Data Engineering) rather
            # than by "azure" (5), which put an Azure question under the
            # broadest subject. "azure key vault" is both longer and the
            # more accurate claim.
            _word("azure", "adls", "key vault", "synapse", "hdinsight",
                  "event hub", "event grid", "stream analytics",
                  "azure storage", "azure functions",
                  "azure key vault", "azure sql database",
                  "azure synapse", "azure databricks",
                  "azure data lake", "google cloud",
                  "service mappings"),
        ),
        subtopics=(
            Subtopic("Storage", (
                _word("adls", "adls gen2", "azure blob", "blob storage",
                      "azure storage", "data lake storage", "hot tier",
                      "cool tier", "archive tier", "gen2", "dfs",
                      "abfss", "file share"),
            )),
            Subtopic("Identity and Secrets", (
                # "What is the role of Azure Key Vault in securing data
                # pipelines?", "How would you implement row-level
                # security and column encryption in Azure SQL Database?"
                _word("securing data pipeline", "key vault in",
                      "row-level security", "column encryption",
                      "access control", "encryption",
                      "key vault", "keyvault", "managed identity",
                      "service principal", "secret", "credential",
                      "azure ad", "authentication", "oauth",
                      "access key", "sas token"),
            )),
            Subtopic("Stream and Messaging", (
                _word("event hub", "event hubs", "event grid",
                      "stream analytics", "kusto", "kql", "service bus",
                      "pub/sub", "real-time ingestion"),
            )),
            Subtopic("Integration Across Azure Services", (
                # "How do you orchestrate workflows across multiple
                # Azure services?", "What approach would you use for
                # incremental and CDC data loading in Azure?"
                _word("incremental and cdc", "cdc data loading",
                      "orchestrate workflow", "across multiple azure",
                      "across azure services", "incremental load in azure",
                      "cdc loading in azure", "azure architecture",
                      "reference architecture", "end-to-end azure",
                      "azure reference", "azure stack",
                      "end-to-end path"),
            )),
            Subtopic("Multi-Cloud Comparison", (
                # "What specific Azure and Google Cloud resources are
                # named in the post?", "What experience do you have with
                # services from Azure, AWS, or GCP?". Shared with AWS and
                # Cloud, which holds the same material from the AWS side;
                # a question naming Azure is placed here, and the
                # material is genuinely about both.
                _word("google cloud", "service mappings", "aws, azure, and gcp",
                      "azure, aws, or gcp", "azure or aws",
                      "multi-cloud", "which cloud"),
            )),
            Subtopic("Analytics and Compute", (
                # "How would you automate backup and restore operations
                # for data stored in Azure SQL Database?", "How would you
                # diagnose and improve the performance of a slow query in
                # Azure SQL Data...", "How would you implement row-level
                # security and column encryption in Azure SQL Database?",
                # "What reporting steps and visual elements are proposed
                # after Azure SQL Database is av..."
                _word("azure sql database", "backup and restore",
                      "slow query in azure", "row-level security",
                      "column encryption", "reporting steps",
                      "sql database", "azure sql", "sql data warehouse",
                      "synapse", "hdinsight", "azure function",
                      "azure functions", "azure batch", "databricks on azure",
                      "vm", "virtual machine", "managed instance",
                      "sql server", "cosmos db", "cosmos"),
            )),
            Subtopic("General", ()),
        ),
    )


def _career() -> Major:
    return Major(
        name="Career and Hiring",
        blurb="How to get through the hiring process: what a round looks "
              "like, and how to prepare. Job announcements and personal "
              "updates live here too -- see the note on Announcements.",
        patterns=(
            _word("job announcement", "hiring", "recruiter", "job post",
                  "job posting", "job search", "career", "resume",
                  "referral", "interview process", "personal announcement",
                  "congratulations", "laid off", "layoff",
                  # "What learning sequence does the post recommend for
                  # someone starting in Data Engineering?", "Why does the
                  # post recommend learning Data Engineering basics
                  # before pursuing flashy tooling?", "How does the post
                  # recommend progressing from Python study to
                  # practical analytics work?". Each names a subject and
                  # then asks about learning it, so the compounds are
                  # longer than the subject words they contain.
                  "learning sequence",
                  "learning data engineering",
                  "starting in data engineering",
                  "progression from python",
                  "practical analytics",
                  "pursuing flashy",
                  "foundation for ai",
                  "solid data engineer",
                  "useful language for beginners",
                  "remain the foundation"),
        ),
        subtopics=(
            # There was a third subtopic here, "Announcements and
            # Updates", and it has been removed after measuring it.
            #
            # 83 posts carry no knowledge label and no interview
            # question: job announcements, personal updates, hiring posts.
            # They are the reason this subject exists -- filing them under
            # a technical subject is what produced "SQL" as the title of
            # an offer announcement, and it is why the catch-all was so
            # full. But a subtopic is only reachable from the revision
            # index if it holds questions, and these posts hold none: the
            # enrichment found no question to ask about them.
            #
            # So they are excluded from the revision curriculum rather
            # than given a dead entry in it. They are not lost. Each still
            # has a knowledge page, titled from its own summary rather
            # than from a technical subject it does not belong to, and all
            # 83 remain in the saved-items list and in search. A revision
            # index is not the right place for an offer announcement, and
            # an empty subtopic is a dead end for exactly the reader who
            # would click it.
            #
            # The subject patterns above still recognise announcement
            # wording, so if such a post ever does carry a question it
            # lands here rather than in a technical subject.
            Subtopic("Interview Process", (
                _word("interview process", "first round", "second round",
                      "hr round", "managerial round", "technical round",
                      "bar-raiser", "bar raiser", "leadership principle",
                      "company-specific", "assessment centre",
                      "assessment center", "online assessment",
                      "aptitude test", "coding round", "how many rounds"),
            )),
            Subtopic("Job Seeking and Preparation", (
                # "What learning sequence does the post recommend for
                # someone starting in Data Engineering?", "Why does the
                # post recommend learning Data Engineering basics before
                # pursuing flashy tooling?", "Which ten technical skill
                # areas does the post identify for AI/ML engineering?",
                # "What technical capabilities does the post identify for a
                # solid Data Engineer?"
                _word("learning sequence", "starting in data engineering",
                      "before pursuing", "skill areas",
                      "technical capabilities does the post",
                      "solid data engineer", "foundation for ai",
                      "before ml and generative",
                      "data engineering capabilities",
                      "foundation remains", "data engineering basics",
                      "what a candidate should know",
                      "should a data engineer know",
                      "demonstrate data engineering ability",
                      "resume", "cv ", "curriculum vitae", "job search",
                      "career advice", "career switch", "referral",
                      "salary negotiation", "notice period", "relocation",
                      "work from home", "remote role", "portfolio",
                      "mentoring", "certification", "roadmap for",
                      "placement", "compensation", "apprenticeship",
                      "internship", "linkedin url", "skills and opportunities",
                      "visibility"),
            )),
            Subtopic("General", ()),
        ),
    )


#: Declared in the order a reader should meet them: the languages, then
#: the platforms, then the wider practice, then the material that is not
#: technical -- last, because it is not what most visits are for.
MAJORS: tuple[Major, ...] = (
    _sql(),
    _python(),
    _spark(),
    _databricks(),
    _adf(),
    _azure_platform(),
    _aws(),
    _data_engineering(),
    _algorithms(),
    _devops(),
    _bi(),
    _machine_learning(),
    _interview(),
    _career(),
)

#: Where a label matching nothing at all is filed. Resolved by name
#: rather than position so that reordering :data:`MAJORS` cannot silently
#: move the catch-all into a different subject.
_FALLBACK = next(m for m in MAJORS if m.slug == "data-engineering")


@dataclass(frozen=True)
class Placement:
    """Where one label was put, and how confidently."""

    major: Major
    subtopic: Subtopic
    matched: str = ""

    @property
    def is_fallback(self) -> bool:
        """True when nothing matched and this is the catch-all."""

        return not self.matched


def _normalise(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip())


def classify(label: str, subjects: tuple[Major, ...] = MAJORS) -> Placement:
    """
    Place one free-text label in the hierarchy.

    Tried in three passes, and the order encodes what a revising reader
    would expect:

    1. every subtopic pattern of every subject, in declaration order, so
       "Spark SQL joins" reaches SQL→Joins rather than whichever subject
       happened to be iterated first;
    2. every subject pattern, choosing the subject's ``general``;
    3. the first subject's ``general``, with the label recorded as
       unmatched so the residual is countable.
    """

    text = _normalise(label)

    if not text:
        return Placement(subjects[0], subjects[0].general())

    # Pass 0: an explicitly named subject wins outright. "Spark SQL
    # joins" is a SQL question asked in Spark, and SQL is declared first,
    # so without this the label would be argued over by two subjects for
    # no benefit.
    #
    # Scored by match length for the same reason pass 1 is. "Spark SQL"
    # contains "sql" (SQL's subject pattern, 3 characters) and "spark
    # sql" (Spark's, 9), and the longer one is the more specific claim;
    # taking the first declared subject instead put every Spark SQL label
    # under SQL and left Spark's own Spark SQL subtopic permanently empty.
    named: tuple[int, Major] | None = None

    for major in subjects:
        for pattern in major.patterns:
            match = pattern.search(text)

            if match is None:
                continue

            length = match.end() - match.start()

            if named is None or length > named[0]:
                named = (length, major)

    if named is not None:
        return _best_in(text, subjects, restrict=named[1])

    # Pass 1: the most specific phrase wins, whatever order the subjects
    # were declared in. "Broadcast joins" matches both SQL's generic "join"
    # and Spark's "broadcast join"; the longer literal is the more
    # informative match, and taking the first subject instead put every
    # broadcast-join label under SQL.
    return _best_in(text, subjects)


def _best_in(
    text: str,
    subjects: tuple[Major, ...],
    *,
    restrict: Major | None = None,
    first: re.Match[str] | None = None,
) -> Placement:
    """
    The most specific subtopic match for a label.

    Scored by how much of the label the pattern actually matched, because
    "broadcast join" is a better answer than "join" for the same label.
    Ties are broken by declaration order, so the outcome is deterministic
    and does not depend on dictionary iteration.
    """

    best: tuple[int, Major, Subtopic, str] | None = None

    scope = (restrict,) if restrict is not None else subjects

    for major in scope:
        for subtopic in major.subtopics:
            for pattern in subtopic.patterns:
                match = pattern.search(text)

                if match is None:
                    continue

                # ``first`` is the subject-level hit that chose this
                # subject; it counts as a full-length match so a
                # subject's own name beats a weak subtopic keyword.
                length = (
                    len(text)
                    if pattern is first
                    else match.end() - match.start()
                )

                if best is None or length > best[0]:
                    best = (length, major, subtopic, pattern.pattern)

    if best is not None:
        return Placement(best[1], best[2], best[3])

    # No subtopic claimed it. Fall back to a subject-level match.
    for major in scope:
        for pattern in major.patterns:
            if pattern.search(text):
                return Placement(major, major.general(), pattern.pattern)

    # Nothing matched anywhere. Data Engineering is the honest catch-all:
    # it is the broadest subject in the taxonomy, so a label about
    # something unrecognised belongs there rather than in whichever
    # subject happened to be declared first. Returning SQL here put 1,505
    # unrecognised labels under "SQL" and made SQL look like two thirds of
    # the corpus.
    fallback = _FALLBACK

    return Placement(fallback, fallback.general())


def major_by_slug(slug: str, subjects: tuple[Major, ...] = MAJORS) -> Major | None:
    for major in subjects:
        if major.slug == slug:
            return major

    return None


def subtopic_by_slug(
    major_slug: str, subtopic_slug: str, subjects: tuple[Major, ...] = MAJORS
) -> tuple[Major, Subtopic] | None:
    major = major_by_slug(major_slug, subjects)

    if major is None:
        return None

    for subtopic in major.subtopics:
        if subtopic.slug == subtopic_slug:
            return major, subtopic

    return None


def slug_of(label: str) -> str:
    """The slug a concept or question is filed under."""

    return _slug(label) or "untitled"


__all__ = [
    "MAJORS",
    "Major",
    "Placement",
    "Subtopic",
    "classify",
    "major_by_slug",
    "slug_of",
    "subtopic_by_slug",
]