#!/bin/bash
set -eo pipefail

# afk.sh — run claude N times unattended, locally on the host (no docker).
#
# Uses --dangerously-skip-permissions so tool calls don't prompt for approval
# while the user is AFK. There is NO sandbox — claude runs against your real
# filesystem with your host credentials, so only invoke this on a workspace
# you're comfortable having edited automatically.
#
# Usage:
#   ./ralph/afk.sh <iterations>
#
# Env vars:
#   MODEL    — model passed to `claude --model` (default: sonnet)
#   FEATURE  — feature folder under .scratch/ (default: simulator-redesign)
#
# Each iteration appends a row to .scratch/$FEATURE/ralph_progress.md so a
# killed / credit-out session can pick up cleanly on next invocation. The
# agent is asked (in prompt.md) to emit a fenced ```report block; this
# script extracts the block and writes one row per iteration. The last
# ~50 lines of the progress file are also fed back into the next iteration's
# context so the agent knows what prior runs did.

if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

MODEL="${MODEL:-sonnet}"
FEATURE="${FEATURE:-simulator-redesign}"
ISSUES_DIR=".scratch/$FEATURE/issues"
PROGRESS_FILE=".scratch/$FEATURE/ralph_progress.md"

# jq filter to extract streaming text from assistant messages
stream_text='select(.type == "assistant").message.content[]? | select(.type == "text").text // empty | gsub("\n"; "\r\n") | . + "\r\n\n"'

# jq filter to extract final result
final_result='select(.type == "result").result // empty'

tmpfile=""
cleanup() { [ -z "$tmpfile" ] || rm -f "$tmpfile"; }
trap cleanup EXIT INT TERM

# Initialize the progress file the first time we run against this feature.
# Only do this if the feature folder exists — otherwise the user has the
# wrong FEATURE set and we should fail loudly inside the loop instead of
# silently materializing the directory.
if [ -d ".scratch/$FEATURE" ] && [ ! -f "$PROGRESS_FILE" ]; then
  init_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "unknown")
  cat > "$PROGRESS_FILE" <<EOF
# Ralph progress — $FEATURE

Branch at start: $init_branch
Started: $(date +"%Y-%m-%d %H:%M:%S")

One row per iteration of \`ralph/afk.sh\`. Status values:

- \`committed\` — task complete, commit landed, issue moved to \`done/\`.
- \`blocked\` — partial work; issue updated to \`needs-info\`.
- \`failed\` — nothing committed, issue left \`ready-for-agent\`.
- \`no-progress\` — iteration produced nothing actionable.
- \`no-tasks\` — no \`ready-for-agent\` issues remain (loop exits).
- \`unknown\` — agent did not emit a parseable report block.

| # | Started | Finished | Issue | Status | Commit | Notes |
|---|---------|----------|-------|--------|--------|-------|
EOF
fi

echo "afk.sh: starting unattended loop for $1 iteration(s) (model=$MODEL feature=$FEATURE)"

for ((i=1; i<=$1; i++)); do
  iter_started_at=$(date +"%Y-%m-%d %H:%M:%S")
  echo ""
  echo "========== Iteration $i/$1 ($iter_started_at) =========="
  echo "afk.sh: collecting context (recent commits, issue files, prior progress, base prompt)"

  tmpfile=$(mktemp)

  commits=$(git log -n 5 --format="%H%n%ad%n%B---" --date=short 2>/dev/null || echo "No commits found")
  issues=$(cat "$ISSUES_DIR"/*.md 2>/dev/null || echo "No issues found")
  progress=$(tail -n 50 "$PROGRESS_FILE" 2>/dev/null || echo "No prior progress")
  prompt=$(cat ralph/prompt.md)

  autonomy=$(cat <<'EOF'

# AUTONOMY

You are running unattended. The user is AFK and cannot answer questions.

- Do NOT ask the user any clarifying questions. Make the best judgment call from the issue file, code, and commit history, and proceed.
- Do NOT use AskUserQuestion or any interactive prompt. Permissions are pre-granted; just run tools directly.
- Treat `.` (the current project root) as the only allowed workspace. Do not read, write, edit, delete, or run commands against paths outside this folder.
- If an issue is genuinely ambiguous and you cannot make a reasonable call, mark its status `needs-info`, add a note explaining what is blocking, and move on to the next `ready-for-agent` issue (or output `<promise>NO MORE TASKS</promise>` if none remain).
- Never block waiting on the user. Decide, document the decision in the commit message, and proceed.
- The "Recent progress" block passed to you is the tail of `ralph_progress.md` — rows from earlier iterations of this same loop. Use it to avoid re-trying issues that already failed or are blocked, and to see what has already shipped.
EOF
)

  echo "afk.sh: launching fresh agent process for iteration $i"
  echo "afk.sh: streaming assistant output below"

  claude \
    --verbose \
    --print \
    --model "$MODEL" \
    --output-format stream-json \
    --dangerously-skip-permissions \
    "Previous commits: $commits Recent progress: $progress Issues: $issues $prompt$autonomy" \
  | tee "$tmpfile" \
  | grep --line-buffered '^{' \
  | jq --unbuffered -rj "$stream_text"
  # Pipeline (tee runs FIRST so the tmpfile captures everything, including
  # non-JSON errors — failures surface instead of being swallowed):
  #   tee "$tmpfile"    → mirror raw stream to disk for post-run inspection
  #   grep '^{'         → keep only JSON lines for jq
  #   jq "$stream_text" → render assistant text live to the terminal

  # If the run produced no JSON at all, claude failed before emitting
  # anything (e.g., not logged in, flag rejected). Surface the raw output.
  if ! grep -q '^{' "$tmpfile"; then
    echo "afk.sh: no JSON received from claude — raw output:" >&2
    cat "$tmpfile" >&2
    echo "" >&2
  fi

  result=$(jq -r "$final_result" "$tmpfile")
  if [ -n "$result" ]; then
    echo "afk.sh: iteration $i result => $result"
  else
    echo "afk.sh: iteration $i finished without final result payload"
  fi

  iter_finished_at=$(date +"%Y-%m-%d %H:%M:%S")

  # Extract the fenced ```report block from the final result text. We use
  # awk to grab the lines between the opening ```report fence and the
  # next closing ``` fence. Anything outside that block is ignored.
  report=$(printf '%s\n' "$result" | awk '
    /^```report[[:space:]]*$/ { flag=1; next }
    /^```[[:space:]]*$/ && flag { flag=0; exit }
    flag { print }
  ')

  # Extracts a single YAML-ish field from the report block. Returns empty
  # string if the field is missing — must NEVER fail the pipeline (which
  # would trip `set -eo pipefail` and skip the row append below).
  extract_field() {
    local field="$1"
    printf '%s\n' "$report" \
      | awk -v key="${field}:" '
          {
            klen = length(key)
            if (substr($0, 1, klen) == key) {
              v = substr($0, klen + 1)
              sub(/^[[:space:]]+/, "", v)
              sub(/^"(.*)"$/, "\\1", v)
              print v
              exit
            }
          }
        '
  }

  rep_issue=$(extract_field "issue_id")
  rep_status=$(extract_field "status")
  rep_commit=$(extract_field "commit_sha")
  rep_notes=$(extract_field "notes")

  : "${rep_issue:=unknown}"
  : "${rep_status:=unknown}"
  : "${rep_commit:=none}"
  : "${rep_notes:=(no report block emitted)}"

  # Short SHA for the table; pass through "none"/"null" untouched.
  case "$rep_commit" in
    none|null|"")    short_sha="none" ;;
    *)               short_sha=$(printf '%s' "$rep_commit" | cut -c1-12) ;;
  esac

  # Escape pipes in notes so they don't break the markdown table; collapse
  # any stray newlines into spaces.
  safe_notes=$(printf '%s' "$rep_notes" | tr '\n' ' ' | sed 's/|/\\|/g')

  if [ -f "$PROGRESS_FILE" ]; then
    printf '| %d | %s | %s | %s | %s | %s | %s |\n' \
      "$i" "$iter_started_at" "$iter_finished_at" "$rep_issue" "$rep_status" "$short_sha" "$safe_notes" \
      >> "$PROGRESS_FILE"
  else
    echo "afk.sh: warning — $PROGRESS_FILE missing; skipping progress row append" >&2
  fi

  rm -f "$tmpfile"
  tmpfile=""

  if [[ "$result" == *"<promise>NO MORE TASKS</promise>"* ]]; then
    echo "afk.sh: Ralph complete after $i iteration(s)."
    exit 0
  fi
done

echo "afk.sh: reached configured iteration limit ($1); stopping."
