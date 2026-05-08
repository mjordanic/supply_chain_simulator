#!/bin/bash
set -eo pipefail

# afk.sh — run claude N times unattended, locally on the host (no docker).
#
# Uses --dangerously-skip-permissions so tool calls don't prompt for approval
# while the user is AFK. There is NO sandbox — claude runs against your real
# filesystem with your host credentials, so only invoke this on a workspace
# you're comfortable having edited automatically.

if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

# jq filter to extract streaming text from assistant messages
stream_text='select(.type == "assistant").message.content[]? | select(.type == "text").text // empty | gsub("\n"; "\r\n") | . + "\r\n\n"'

# jq filter to extract final result
final_result='select(.type == "result").result // empty'

tmpfile=""
cleanup() { [ -n "$tmpfile" ] && rm -f "$tmpfile"; }
trap cleanup EXIT INT TERM

echo "afk.sh: starting unattended loop for $1 iteration(s)"

for ((i=1; i<=$1; i++)); do
  iter_started_at=$(date +"%Y-%m-%d %H:%M:%S")
  echo ""
  echo "========== Iteration $i/$1 ($iter_started_at) =========="
  echo "afk.sh: collecting context (recent commits, issue files, base prompt)"

  tmpfile=$(mktemp)

  commits=$(git log -n 5 --format="%H%n%ad%n%B---" --date=short 2>/dev/null || echo "No commits found")
  issues=$(cat .scratch/simulator-redesign/issues/*.md 2>/dev/null || echo "No issues found")
  prompt=$(cat ralph/prompt.md)

  autonomy=$(cat <<'EOF'

# AUTONOMY

You are running unattended. The user is AFK and cannot answer questions.

- Do NOT ask the user any clarifying questions. Make the best judgment call from the issue file, code, and commit history, and proceed.
- Do NOT use AskUserQuestion or any interactive prompt. Permissions are pre-granted; just run tools directly.
- Treat `.` (the current project root) as the only allowed workspace. Do not read, write, edit, delete, or run commands against paths outside this folder.
- If an issue is genuinely ambiguous and you cannot make a reasonable call, mark its status `needs-info`, add a note explaining what is blocking, and move on to the next `ready-for-agent` issue (or output `<promise>NO MORE TASKS</promise>` if none remain).
- Never block waiting on the user. Decide, document the decision in the commit message, and proceed.
EOF
)

  echo "afk.sh: launching fresh agent process for iteration $i"
  echo "afk.sh: streaming assistant output below"

  claude \
    --verbose \
    --print \
    --output-format stream-json \
    --dangerously-skip-permissions \
    "Previous commits: $commits Issues: $issues $prompt$autonomy" \
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

  rm -f "$tmpfile"
  tmpfile=""

  if [[ "$result" == *"<promise>NO MORE TASKS</promise>"* ]]; then
    echo "afk.sh: Ralph complete after $i iteration(s)."
    exit 0
  fi
done

echo "afk.sh: reached configured iteration limit ($1); stopping."
