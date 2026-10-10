# Contributing to flex-pse

Thanks for your interest in contributing! This project welcomes issues, pull requests, and discussion from anyone. To keep things maintainable and trustworthy for everyone, please read through these guidelines before submitting.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Human Readability Standards](#human-readability-standards)
- [Contributor Responsibility](#contributor-responsibility)
- [Use of AI Tools](#use-of-ai-tools)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Config schema versioning](#config-schema-versioning)
- [Review & Verification by Maintainers](#review--verification-by-maintainers)
- [Code of Conduct](#code-of-conduct)
- [Reporting Bugs & Security Issues](#reporting-bugs--security-issues)
- [Licensing](#licensing)

---

## Ways to Contribute

- **Bug reports** — open an issue with clear reproduction steps.
- **Feature requests** — open an issue describing the problem, not just the solution.
- **Code contributions** — via pull request (see below).
- **Documentation** — fixes, clarifications, and examples are always welcome.
- **Reviewing open PRs/issues** — thoughtful feedback from the community speeds everything up.

## Getting Started

1. Fork the repository and clone it locally.
2. Install dependencies.
3. Run the test suite to confirm a clean baseline.
4. Create a branch for your work: `git checkout -b your-short-branch-name`

## Human Readability Standards

Code and documentation in this repository are read far more often than they are written. All contributions must be **understandable by a human reviewer without special tooling**. Specifically:

- **Clear naming.** Variables, functions, and files should be named for what they do, not abbreviated cryptically.
- **Comments explain "why," not "what."** Don't restate the code in prose; explain intent, trade-offs, and non-obvious decisions.
- **Reasonable diffs.** Keep pull requests focused on a single logical change. Large, unrelated changes bundled together will be asked to be split up.
- **Consistent style.** Follow the existing formatting/linting configuration in this repo. Run it before submitting.
- **Docs and comments must read as if a person wrote them for another person** — plain, direct language, not padded or filler text.

## Contributor Responsibility
By submitting a contribution, **you are personally vouching for it**. Specifically, you confirm that:

- You wrote or meaningfully reviewed the content — you are not simply relaying output you don't understand and asking maintainers to check it for you.
- You have tested the change yourself, locally, before opening the PR.
- You have the right to submit the contribution (it's your own work, or appropriately licensed/attributed).
- You will respond to review feedback on your own submission. Contributions that go unanswered for 10 days after requested changes may be closed.

Maintainers are volunteers reviewing in good faith. Submitting content you can't explain or defend shifts your verification burden onto them, which isn't a fair trade and will typically result in the PR being closed.

## Use of AI Tools

AI-assisted contributions (code, docs, or issue text) are welcome, but the same responsibility bar applies as above — the tool doesn't lower it.

- **Disclose it.** If a substantial part of a PR or issue was generated or drafted with an AI tool, say so in the PR description (a short note is enough — no need to detail every prompt).
- **You must still understand and verify it.** Review, test, and be prepared to explain any AI-assisted content as if you wrote it yourself. "The AI generated it" is not an acceptable answer to a review question.
- **No unreviewed bulk submissions.** Large batches of AI-generated issues or PRs that haven't been individually vetted by a human will be closed without full review.
- **Fabricated content is not acceptable.** Do not submit AI-generated citations, benchmarks, test results, or changelog entries that you haven't personally verified against reality.

## Submitting a Pull Request

1. Ensure your branch is up to date with `main`.
2. Include a clear PR description: what changed, why, and how you tested it.
3. Link any related issue(s).
4. Make sure CI passes before requesting review.
5. Keep the scope of the PR narrow — one concern per PR.

## Config schema versioning

- An additive, optional field is a patch bump with no migration.
- A renamed, reshaped, newly required, or newly rejected field is a bump with a `MIGRATIONS` hook in the same PR, plus a fixture `fixtures/configs/<old version>.json`.
- A migration that cannot translate raises `FlexConfigError` naming the field and how to fix it.
- Regenerate `schemas/model_config.schema.json` in the same PR.
- There is one schema, the flat `FlowsheetSpec`, versioned by `SCHEMA_VERSION`. Versions up to 0.0.4 are the old nested `ModelConfig`; the `0.0.4` entry in `MIGRATIONS` converts them to the flat spec. Keep a fixture `fixtures/specs/<old version>.json` for each flat version and regenerate `schemas/flowsheet_spec.schema.json`.
- A new element kind must be added to `KINDS` with its stage and mutability.

## Review & Verification by Maintainers

All contributions are subject to maintainer review before merging. This is a verification step, not a rubber stamp:

- **Maintainer approval** is required before merge.
- Maintainers may request changes, ask clarifying questions, or ask the contributor to explain specific parts of the change — please respond rather than resubmitting silently.
- Automated checks (tests, linting, security scanning) must pass, but passing CI does not guarantee merge — human review of correctness, readability, and intent is still required.
- Maintainers reserve the right to decline contributions that don't meet the standards above, even if functionally correct, if they harm long-term readability or maintainability.

## Code of Conduct

This project follows a Code of Conduct. Please be respectful and constructive in issues, PRs, and discussions. 

## Reporting Bugs & Security Issues

- Regular bugs: open a GitHub issue with reproduction steps.
- Security vulnerabilities: do **not** open a public issue. Instead, email `akshayr135 at gmail dot com` or follow the process in `SECURITY.md`.

## Licensing

By contributing, you agree that your contributions will be licensed under this project's Apache 2.0 license (see `LICENSE`).

---

Questions not covered here? Open a discussion or reach out to the maintainers at `akshayr135 at gmail dot com`.
