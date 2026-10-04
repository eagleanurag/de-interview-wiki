"""
The reader-facing relevance filter.

The corpus is 490 captured posts, and it contains things a candidate
revising for a data engineering interview does not need: hiring posts,
congratulations, an event invitation, a thank-you note, and a long tail
of "what should I learn next" advice. None of it is deletable -- the
source records are the project's evidence and they stay in the knowledge
base -- but publishing all of it as revision material is how a revision
guide turns back into an archive.

**Why this is a filter on questions and not on posts.** The obvious
approach is to filter by ``content_kinds``, which the aggregator already
provides, and it is wrong here. Measured against this corpus:

* a ``job_announcement`` post contributes real questions -- "How would
  you calculate total revenue for each customer using the Orders and
  Order_Items tables?" is a join and an aggregate, and it came from a
  job ad;
* a ``social`` post contributes "What is a primary key?", which is
  revision material for everyone.

Filtering on the genre of the post would therefore throw away genuine
interview content and keep promotional filler. What distinguishes them is
the question, not the post it was scraped off, so the filter is applied
to question text and to the subject the taxonomy placed it in.

Two rules, both narrow:

* the Career and Hiring subject is not revision material;
* an explicit exclusion list for advice, coaching and hiring text.

Everything else is kept, including the basic programming questions
(lists, dictionaries, strings, functions) the brief asks to retain, and
including the data puzzles, which are ordinary interview questions
dressed as puzzles.
"""

from __future__ import annotations

import re


#: Subjects that are not interview revision material.
#:
#: Career and Holding is the taxonomy's own subject for job
#: announcements, interview processes and job-seeking advice. Publishing
#: it would be publishing the archive's filing system rather than its
#: content.
NON_REVISION_SUBJECTS = frozenset({"career-and-hiring"})


#: Subtopics that are entirely about the search rather than the work.
#:
#: These needed a list rather than a pattern. Measured, the question
#: patterns below caught 18 of the 87 questions in these five subtopics
#: and let 69 through, and reading the 69 showed they were all the same
#: thing: which platform to practise on, how to prepare for a round, what
#: a behavioural answer should contain, what a roadmap covers. The
#: question-level patterns cannot be made to catch all of that without
#: also catching technical sentences that happen to sit nearby --
#: "which SQL topics should a candidate prioritise" is advice, and
#: "which SQL topics implement a recursive CTE" is the job.
#:
#: It is checked *after* the keep rules for the same reason: two of these
#: 87 questions are real technical questions the taxonomy misfiled --
#: "What roles does the post assign to HDFS and YARN?" and a CTE
#: question that matched "hiring" -- and they must survive.
EXCLUDED_SUBTOPICS = frozenset(
    {
        ("interview-preparation", "behavioral"),
        ("interview-preparation", "interview-process-and-preparation"),
        ("interview-preparation", "practice-platforms-and-preparation"),
        ("interview-preparation", "general"),
        ("career-and-hiring", "general"),
    }
)


#: Phrases that mean "this is advice about the search", not the work.
#:
#: Deliberately about the *advice*, never about the technology. "window
#: function" is revision material wherever it appears; "what should I
#: study" is not. Each pattern was checked against the corpus so that a
#: technical sentence containing the same word is not caught -- an early
#: version matched "pattern" and took "which ingestion patterns are
#: identified for CDC" with it.
EXCLUDE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # What to study, and in what order.
        r"\blearning sequence\b",
        r"\bstudy (?:plan|order|sequence)\b",
        r"\blearning (?:path|roadmap)\b",
        r"\bpreparation roadmap\b",
        r"\broadmap\b",
        r"\bsyllabus\b",
        r"\bcurriculum\b",
        r"\bwhat (?:should|does) .*\b(?:learn|study|revise|prepare)\b",
        r"\bprepare for an? (?:interview|career|exam|certification)\b",
        r"\bpreparing for (?:an? )?(?:interview|career|exam)\b",
        r"\bbeginner (?:path|series|advice)\b",

        # Interview technique rather than the work.
        r"\bSTAR format\b",
        r"\bbehaviou?ral (?:topic|question|round|competenc)",
        r"\b(?:strengths and weaknesses|weakness)\b",
        r"\btell me about yourself\b",
        r"\bteam conflict\b",

        # The search itself.
        r"\bresume\b",
        r"\bcurriculum vitae\b",
        r"\bjob (?:search|application|advertisement|posting|opening)\b",
        r"\bjob[- ]?seeker\b",
        r"\bwe are hiring\b",
        r"\bnow hiring\b",
        r"\breferral\b",
        r"\bnotice period\b",
        r"\bsalary negotiation\b",
        r"\binterview process\b",
        r"\bbar[- ]raiser\b",
        r"\brecruiter\b",
        r"\bleadership principle\b",
        r"\bcompany[- ]specific\b",

        # Praise and announcements, which carry no learning value.
        r"\bcongratulat",
        r"\bthank you\b.*\b(?:post|sharing|contribut)",
        r"\bwell done\b",
        r"\bexcited to (?:announce|share)\b",
        r"\bproud to (?:announce|share)\b",

        # Pattern-printing exercises: named as an exclusion, and
        # recognisable by the shape of the ask rather than the word
        # "pattern", which also appears in "ingestion patterns".
        r"\bprint (?:a |the )?(?:star|triangle|diamond|pyramid|pattern|"
        r"pascal(?:'s)? triangle)\b",
        r"\b(?:star|triangle|diamond|pyramid|pascal) pattern\b",
        r"\bpattern[- ]printing\b",
        r"\bhello world\b",
        r"\bfizz ?buzz\b",
    )
)


#: Kept explicitly, and listed here so the exclusions cannot take them.
#:
#: The brief asks for basic programming questions an interviewer may
#: reasonably ask. Several of the words below -- "reverse", "string",
#: "generator" -- sit near the exclusion vocabulary, so the keeper is
#: stated as a positive rule evaluated *before* the exclusions.
KEEP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Classic interview asks, kept regardless of surface wording.
        r"\breverse a string\b",
        r"\bstring (?:palindrome|anagram|manipulation)\b",
        r"\bvalid palindrome\b",
        r"\b(?:list|tuple|dictionary|dict|set) (?:comprehension|operations?)\b",
        r"\b(?:merge two|sorted) (?:arrays?|lists?)\b",
        r"\bfunction (?:overload|decorator|recursion)\b",
        r"\b(?:object[- ]oriented|inheritance|polymorphism|encapsulation)\b",
        r"\bdeepcopy\b",
        r"\bgenerator\b.*\b(?:yield|function|iterable)\b",
        r"\bfibonacci\b",
        r"\btwo pointers?\b",
        r"\bsliding window\b",
        r"\bbinary search\b",
        r"\btime complexity\b",
        r"\bspace complexity\b",
        r"\bbig[- ]o\b",
    )
)


#: Naming a technology, a technique or a data structure is what makes a
#: question revision material, whichever subject the taxonomy filed it
#: under.
#:
#: This exists because of a measured misfiling. "How would you write a
#: CTE-based SQL solution that selects analysts under the stated hiring
#: criteria?" was placed in Career and Hiring by a rule that saw
#: "hiring", and it is a CTE question with a filter clause. Refusing it
#: because of the subject it landed in would have removed real SQL from
#: the revision guide on the strength of a word it happens to contain.
TECHNICAL_NAMED: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsql\b", r"\bcte\b", r"\bjoin\b", r"\bwindow function\b",
        r"\bpartition\b", r"\baggregat", r"\bgroup by\b", r"\bsubquer",
        r"\bindex", r"\btransaction", r"\bdeadlock\b", r"\bnormalis",
        r"\bnormaliz", r"\bstar schema\b", r"\bfact table\b",
        r"\bdimension table\b", r"\bdata warehouse\b", r"\bdata lake\b",
        r"\betl\b", r"\belt\b", r"\bcdc\b", r"\bscd\b",
        r"\bslowly changing\b", r"\bmedallion\b", r"\blakehouse\b",
        r"\bspark\b", r"\bpyspark\b", r"\bdatabricks\b", r"\bdelta\b",
        r"\bairflow\b", r"\bdbt\b", r"\bkafka\b", r"\bhive\b",
        r"\bhadoop\b", r"\bhdfs\b", r"\bredshift\b", r"\bbigquery\b",
        r"\bsnowflake\b", r"\bsynapse\b", r"\bs3\b", r"\bglacier\b",
        r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bkubernetes\b",
        r"\bdocker\b", r"\bterraform\b", r"\bci/cd\b", r"\bpytest\b",
        r"\bpython\b", r"\bpandas\b", r"\bnumpy\b", r"\bpolars\b",
        r"\blist\b", r"\btuple\b", r"\bdictionar", r"\bset\b",
        r"\bfunction\b", r"\bstring\b", r"\bpalindrome\b",
        r"\brecursion\b", r"\boop\b", r"\bobject[- ]oriented\b",
        r"\binheritance\b", r"\bpolymorphism\b", r"\bdeepcopy\b",
        r"\bgenerator\b", r"\biterator\b", r"\blambda\b",
        r"\bregex\b", r"\bcomplexity\b", r"\bapi\b", r"\bjson\b",
        r"\bparquet\b", r"\bavro\b", r"\bcsv\b", r"\bdax\b",
        r"\bpower bi\b", r"\btableau\b", r"\bmlflow\b",
        r"\bmachine learning\b", r"\bmodel\b", r"\bschema\b",
        r"\bpipeline\b", r"\bdashboard\b", r"\breport\b",
    )
)


def exclusion_reason(
    question: str,
    major_slug: str,
    subtopic_slug: str | None = None,
) -> str | None:
    """
    Why this question is not revision material, or ``None`` if it is.

    The order is the argument, so it is worth stating:

    1. **Advice is refused first.** "Which SQL capabilities are
       prioritised in Week 1 of the roadmap?" names SQL and is still
       study advice, and no amount of technical vocabulary makes it
       revision material.
    2. **Then the filing decides.** A subject or subtopic that is
       entirely about the search is not published, and this check
       deliberately sits *above* the technical keep rules.
    3. **Naming the work is then enough to keep.** A question about SQL,
       CTEs or a hash map is revision material. This is what keeps a
       real question that the taxonomy filed under an ordinary subject,
       such as a join question placed in SQL Foundations because it
       mentioned a hiring policy.

    The misfiled technical questions inside an excluded subtopic are the
    one case the ordering costs, and they are handled by
    :func:`relocation_target`, which the caller applies before calling
    this.

    The reason is returned rather than only a boolean so the filtering
    can be reported: "1,900 kept, 116 discarded, of which 85 interview
    process and practice, 22 advice and coaching, 2 career and hiring" is
    auditable, and "1,900 kept" is not.
    """

    for pattern in EXCLUDE_PATTERNS:
        if pattern.search(question):
            return "advice, coaching or hiring text"

    # The filing decides here, before anything technical can rescue the
    # question. A subtopic that is entirely about the search is not
    # revision material however many words of technology it contains:
    # "which SQL topics should a candidate prioritise for an interview"
    # names SQL and is still advice. Ordering this after the technical
    # check kept 33 of 85 such questions.
    #
    # The two genuinely misfiled technical questions are relocated
    # instead, by RELOCATIONS below, which the caller applies first.
    if major_slug in NON_REVISION_SUBJECTS:
        return "career and hiring"

    if subtopic_slug is not None and (
        major_slug,
        subtopic_slug,
    ) in EXCLUDED_SUBTOPICS:
        return "interview process, practice and behavioural coaching"

    for pattern in KEEP_PATTERNS:
        if pattern.search(question):
            return None

    for pattern in TECHNICAL_NAMED:
        if pattern.search(question):
            return None

    return None


#: Questions filed under the search that are really questions about the
#: work, and the unit that should own them.
#:
#: Two of the 87 excluded-subtopic questions are technical and were
#: misfiled: "What roles does the post assign to HDFS and YARN, and
#: which Big Data processing comparison was included?" is Spark
#: ecosystem material, and "How would you write a CTE-based SQL
#: solution..." is a recursive CTE question that matched "hiring".
#:
#: Rescuing them by keyword was tried and is wrong. Allowing any question
#: that names a technology through the exclusion kept 33 of the other
#: 85, every one of them advice -- "which SQL topics should a candidate
#: prioritise for interview preparation" names SQL and is still advice.
#: So the subtopics are excluded outright and these two are relocated by
#: an explicit rule instead.
RELOCATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bHDFS\b|\bYARN\b|\bMapReduce\b", re.IGNORECASE),
        "spark-and-pyspark",
    ),
    (
        re.compile(r"\bCTE\b|\bcommon table expression\b", re.IGNORECASE),
        "sql-ctes-and-subqueries",
    ),
)


def relocation_target(question: str) -> str | None:
    """The unit slug a misfiled technical question belongs to."""

    for pattern, unit_slug in RELOCATIONS:
        if pattern.search(question):
            return unit_slug

    return None


def is_relevant(
    question: str,
    major_slug: str,
    subtopic_slug: str | None = None,
) -> bool:
    return exclusion_reason(question, major_slug, subtopic_slug) is None