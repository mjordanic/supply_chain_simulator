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
#   AFK_PRINT_INPUT — set to `1` to dump commits, issues, prompt, and
#                     autonomy to stderr before each claude invocation
#
# Cross-iteration context comes from git commits and the issue files
# themselves — this script does NOT pre-load past progress into the
# agent's context window. The agent is handed the path to a progress log
# (.scratch/$FEATURE/ralph_progress.md) and is expected to append a row
# to it at the end of each iteration as a human-readable audit trail.
#
# CRASH RECOVERY PROTOCOL
# -----------------------
# The progress log doubles as a crash-detection trip-wire. Per iteration:
#
#   1. (script, pre-invocation)  If the file's last line is a sentinel
#      left by a previous iteration, that iteration died before the
#      agent could append its row. Strip the stale sentinel and inject a
#      `# RECOVERY` block into THIS iteration's prompt, telling the agent
#      to inspect `git status` / `git diff HEAD` before doing anything
#      else (the working tree may hold half-finished changes).
#
#   2. (script, pre-invocation)  Append a fresh sentinel marking this
#      iteration's start. This both arms the trip-wire for the next
#      iteration AND creates the progress file on first-ever run.
#
#   3. (agent, during run)       Do the work; as the last step of its
#      REPORT phase, append a markdown row to the file. The agent never
#      touches the sentinel — that's the script's responsibility.
#
#   4. (script, post-invocation) If the sentinel is no longer the last
#      line of the file, the agent appended its row → clean exit, strip
#      the sentinel. If the sentinel IS still the last line, the agent
#      crashed/aborted → leave it so step 1 of the next iteration fires.
#
# Invariant between iterations: the file ends with EITHER the agent's
# most recent row (clean state) OR an in-progress sentinel (crashed
# state). Successful iterations remove the sentinel they wrote; only
# crashes leave one behind, so the file does not accumulate dust.
#
# Sentinel format: a single HTML-comment line beginning with
# `<!-- afk.sh:in-progress` — invisible in rendered markdown, trivial to
# grep for, distinguishable from any markdown table row the agent writes.

if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

MODEL="${MODEL:-sonnet}"
FEATURE="${FEATURE:-simulator-redesign}"
ISSUES_DIR=".scratch/$FEATURE/issues"
PROGRESS_FILE=".scratch/$FEATURE/ralph_progress.md"
# Crash-recovery sentinel — see "CRASH RECOVERY PROTOCOL" at top of file.
# Any line beginning with this prefix is owned by afk.sh and stripped on
# detection; the agent is told (via prompt.md) to leave such lines alone.
SENTINEL_PREFIX="<!-- afk.sh:in-progress"

# jq filter to extract streaming text from assistant messages
stream_text='select(.type == "assistant").message.content[]? | select(.type == "text").text // empty | gsub("\n"; "\r\n") | . + "\r\n\n"'

# jq filter to extract final result
final_result='select(.type == "result").result // empty'

tmpfile=""
cleanup() { [ -z "$tmpfile" ] || rm -f "$tmpfile"; }
trap cleanup EXIT INT TERM

afk_print_section() {
  printf 'afk.sh: === %s ===\n' "$1" >&2
  printf '%s\n' "$2" >&2
  echo >&2
}

echo "afk.sh: starting unattended loop for $1 iteration(s) (model=$MODEL feature=$FEATURE)"

for ((i=1; i<=$1; i++)); do
  iter_started_at=$(date +"%Y-%m-%d %H:%M:%S")
  echo ""
  echo "========== Iteration $i/$1 ($iter_started_at) =========="
  echo "afk.sh: collecting context (recent commits, issue files, base prompt)"

  tmpfile=$(mktemp)

  commits=$(git log -n 5 --format="%H%n%ad%n%B---" --date=short 2>/dev/null || echo "No commits found")
  issues=$(cat "$ISSUES_DIR"/*.md 2>/dev/null || echo "No issues found")
  prompt=$(cat ralph/prompt.md)

  # ---- Recovery protocol step 1: detect a stale sentinel. ----
  # The post-invocation cleanup (step 4, below the claude call) only
  # strips the sentinel when the agent appended a row after it. So if
  # the file's LAST line is still our sentinel, the previous iteration
  # never reached "agent appended row" — it crashed, was SIGKILLed, the
  # machine rebooted, etc. Strip the stale sentinel and build a RECOVERY
  # block to prepend to this iteration's prompt; the agent will see it
  # and inspect the working tree before doing anything else.
  recovery_note=""
  if [ -f "$PROGRESS_FILE" ] && tail -n1 "$PROGRESS_FILE" 2>/dev/null | grep -q "^$SENTINEL_PREFIX"; then
    # awk filter (idempotent): keep every line that does NOT start with
    # the sentinel prefix. Removes both the immediate stale sentinel and
    # any orphans left by past abnormal exits.
    awk -v pfx="$SENTINEL_PREFIX" 'index($0, pfx) != 1 { print }' "$PROGRESS_FILE" > "$PROGRESS_FILE.tmp" \
      && mv "$PROGRESS_FILE.tmp" "$PROGRESS_FILE"
    # GOTCHA: do not rewrite this as `recovery_note=$(cat <<'EOF' ... EOF)`.
    # Bash's `$()` parser pre-scans the heredoc body and an apostrophe
    # in the text (e.g. "doesn't") opens a phantom quote it never
    # closes, breaking the whole script with a misleading "unexpected
    # EOF" error. `IFS= read -r -d ''` sidesteps the parser entirely;
    # `IFS=` preserves the leading blank lines that visually separate
    # this block from `$autonomy` when both are concatenated into the
    # final prompt. The `|| true` swallows read's nonzero exit (it
    # always returns nonzero when the delimiter, NUL here, isn't found).
    IFS= read -r -d '' recovery_note <<'EOF' || true


# RECOVERY

The previous afk.sh iteration did not finish cleanly — its in-progress
marker was still present in `__PROGRESS_FILE__`. The working tree may
contain uncommitted partial changes from that run. Before picking a new
task:

1. Run `git status` and `git diff HEAD` to see leftover changes.
2. Decide deliberately based on what you find:
   - Changes coherent and aligned with an in-progress issue → finish + commit.
   - Changes partial or unclear → run `git restore` (and `git clean -fd`
     for untracked files) to discard them and start fresh.
3. Document the decision in the next commit message or issue notes, then
   proceed with normal task selection.
EOF
    recovery_note=${recovery_note//__PROGRESS_FILE__/$PROGRESS_FILE}
    echo "afk.sh: detected stale sentinel — injecting RECOVERY block into prompt"
  fi

  # ---- Recovery protocol step 2: arm the trip-wire. ----
  # Append a sentinel marking THIS iteration's start. If we crash
  # between here and the agent's row append, step 1 of the NEXT
  # iteration will see this line as the file's tail and trigger
  # recovery. Side effect: on the very first run for a feature, this
  # `>>` creates the progress file. The agent's prompt instructs it to
  # prepend a markdown table header before its first row, so the file
  # ends up well-formed without the script writing a header itself.
  echo "$SENTINEL_PREFIX (started $iter_started_at) -->" >> "$PROGRESS_FILE"

  # Same `IFS= read -r -d ''` heredoc pattern as recovery_note above —
  # see the GOTCHA comment there. Apostrophes in this block (e.g.
  # "doesn't exist") would break a `$(cat <<EOF … EOF)` form.
  IFS= read -r -d '' autonomy <<'EOF' || true

# AUTONOMY

You are running unattended. The user is AFK and cannot answer questions.

- Do NOT ask the user any clarifying questions. Make the best judgment call from the issue file, code, and commit history, and proceed.
- Do NOT use AskUserQuestion or any interactive prompt. Permissions are pre-granted; just run tools directly.
- Treat `.` (the current project root) as the only allowed workspace. Do not read, write, edit, delete, or run commands against paths outside this folder.
- If an issue is genuinely ambiguous and you cannot make a reasonable call, mark its status `needs-info`, add a note explaining what is blocking, and move on to the next `ready-for-agent` issue (or output `<promise>NO MORE TASKS</promise>` if none remain).
- Never block waiting on the user. Decide, document the decision in the commit message, and proceed.
- Progress log path: `__PROGRESS_FILE__`. Append a row at the end of your iteration (create the file with a header if it doesn't exist). Use git commits as the source of truth for what prior iterations did.
EOF
  autonomy=${autonomy//__PROGRESS_FILE__/$PROGRESS_FILE}

  if [ "${AFK_PRINT_INPUT:-}" = "1" ]; then
    echo "afk.sh: AFK_PRINT_INPUT=1 — values embedded in the claude user message (stderr)" >&2
    echo "afk.sh: --model $MODEL" >&2
    echo >&2
    afk_print_section "commits (git log)" "$commits"
    afk_print_section "issues (*.md under issues/)" "$issues"
    afk_print_section "prompt (ralph/prompt.md)" "$prompt"
    afk_print_section "autonomy (appended by afk.sh)" "$autonomy"
    if [ -n "$recovery_note" ]; then
      afk_print_section "recovery (stale sentinel detected)" "$recovery_note"
    fi
    echo "afk.sh: assembled as: \"Previous commits:\" + commits + \" Issues:\" + issues + prompt + autonomy + recovery_note" >&2
    echo >&2
  fi

  echo "afk.sh: launching fresh agent process for iteration $i"
  echo "afk.sh: streaming assistant output below"

  claude \
    --verbose \
    --print \
    --model "$MODEL" \
    --output-format stream-json \
    --dangerously-skip-permissions \
    "Previous commits: $commits Issues: $issues $prompt$autonomy$recovery_note" \
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

  # ---- Recovery protocol step 4: sentinel cleanup. ----
  # The agent's row append (step 3) is the signal that it finished
  # cleanly. Read the file's last line:
  #   - Last line is NOT the sentinel → the agent appended a row after
  #     it. Strip the sentinel; the file now ends with the agent's row,
  #     restoring the clean-state invariant.
  #   - Last line IS the sentinel → the agent never appended (mid-run
  #     crash, no result emitted, agent ignored REPORT instructions).
  #     Leave the sentinel; step 1 of the NEXT iteration will detect it
  #     and trigger recovery.
  # The trip-wire only fires when there's something genuinely worth
  # recovering: the agent did not complete its REPORT.
  if [ -f "$PROGRESS_FILE" ] && ! tail -n1 "$PROGRESS_FILE" 2>/dev/null | grep -q "^$SENTINEL_PREFIX"; then
    awk -v pfx="$SENTINEL_PREFIX" 'index($0, pfx) != 1 { print }' "$PROGRESS_FILE" > "$PROGRESS_FILE.tmp" \
      && mv "$PROGRESS_FILE.tmp" "$PROGRESS_FILE"
  fi

  if [[ "$result" == *"<promise>NO MORE TASKS</promise>"* ]]; then
    echo "afk.sh: Ralph complete after $i iteration(s)."
    exit 0
  fi
done

echo "afk.sh: reached configured iteration limit ($1); stopping."
