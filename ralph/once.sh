#!/bin/bash
# once.sh — run a SINGLE ralph iteration interactively (acceptEdits, no sandbox).
# Good for trying the loop on one task before kicking off an unattended afk.sh run.
#
# Usage:  ./ralph/once.sh
# Env:    FEATURE — feature folder under .scratch/ (default: simulator-redesign)

FEATURE="${FEATURE:-simulator-redesign}"
issues=$(cat ".scratch/$FEATURE/issues"/*.md 2>/dev/null || echo "No issues found")
commits=$(git log -n 5 --format="%H%n%ad%n%B---" --date=short 2>/dev/null || echo "No commits found")
prompt=$(cat ralph/prompt.md)

claude --permission-mode acceptEdits \
  "Previous commits: $commits Issues: $issues $prompt"
