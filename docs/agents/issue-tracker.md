# Issue tracker: Local Markdown

Issues and PRDs for this repo live as markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The PRD is `.scratch/<feature-slug>/PRD.md`
- Implementation issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state is recorded as a `Status:` line near the top of each issue file (see `triage-labels.md` for the role strings)
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

## Quality checklist for PRDs and issues

Apply when writing a PRD (`/to-prd`), breaking one into issues (`/to-issues`), or reviewing either:

- **Story coverage (inverse check)** — every PRD user story is owned by at least one issue, or explicitly noted as satisfied-by-property (name the issue and the property) or out of scope. No unowned stories.
- **Artifact homes** — every artifact a PRD or issue mandates (README, report, emitted data) gets an explicit on-disk location decision. If the artifact is meant to be committed, verify the path is trackable (`git check-ignore <path>`) — this repo blanket-ignores `data/`, so a committed file placed there would be silently untracked.
- **Prior-art pointers** — each issue carries pointers to relevant prior art (especially test prior art) using stable names (test names, module names, glossary terms), not file paths.
- **PRD ↔ issue sync** — when review edits an issue (scope, artifact locations, decisions), update the PRD to match. The two documents must not disagree when the implementer reads them.
