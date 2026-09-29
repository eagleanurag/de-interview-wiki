"""
Remote OpenCode control plane.

This package contains the deterministic, unit-testable logic that the
`.github/workflows/opencode-agent.yml` workflow relies on:

* `events`   - authorize actors, detect triggers, extract the task
* `prompt`   - build the autonomous task-contract prompt
* `opencode` - run the CLI and parse its JSON event stream
* `ci`       - trigger, await and inspect the validation pipeline
* `reporting`- build issue reports and redact anything sensitive

The workflow stays declarative YAML; everything with a decision in it
lives here so it can be tested without a GitHub runner.
"""
