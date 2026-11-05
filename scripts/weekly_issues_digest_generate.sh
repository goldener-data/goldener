#!/usr/bin/env bash
# Generate a Slack digest JSON file using only GitHub CLI and its formatter.
# Usage: REPO=owner/repo bash scripts/weekly_issues_digest_generate.sh <digest.json>
# Requires Bash and authenticated gh. See weekly_issues_digest.md.
# Stop on command failures, unset variables, or failures inside pipelines.
set -euo pipefail

if [[ $# != 1 ]]; then
    echo 'Usage: bash scripts/weekly_issues_digest_generate.sh <digest.json>' >&2
    exit 1
fi
REPO=${REPO:-goldener-data/goldener}

# Fetch every open ready issue; gh handles pagination internally. The large limit
# avoids the default 30-issue cap. The --jq expression below builds the saved JSON.
gh issue list --repo "$REPO" --state open --label ready --limit 2147483647 \
  --json number,title,url,createdAt,labels --jq '
  # Escape Slack control characters so issue titles cannot create mentions.
  def escape: gsub("&"; "&amp;") | gsub("<"; "&lt;") | gsub(">"; "&gt;");
  # Shorten escaped text without leaving a partial HTML entity at the end.
  def shorten($limit):
    escape | if length > $limit then .[:$limit] | sub("&[^;]*$"; "") + "…" else . end;
  # Remove decorative punctuation and emoji around skill label values.
  def strip_decor: gsub("^[^\\p{L}\\p{N}]+|[^\\p{L}\\p{N}]+$"; "");
  # Use the first matching skill label: "skill: beginner" becomes "beginner".
  # Keep named labels such as "good first issue" when removing the match leaves no value.
  def skill:
    [.labels[].name | select(test("skill|level|good first issue|help wanted"; "i"))][0] as $label |
    if $label == null then "unspecified"
    else ($label | sub("skill|level|good first issue|help wanted"; ""; "i") | strip_decor) as $value |
      if $value == "" then $label | strip_decor else $value end
    end;
  def emoji:
    if . == "good first issue" then "🟢"
    elif . == "help wanted" then "🟡"
    elif . == "unspecified" then "⚪" else "🔹" end;
  # Normalize GitHub field names; count also tells the workflow whether to send.
  map({number, title, html_url: .url, created_at: .createdAt, labels}) as $issues |
  if ($issues | length) == 0 then {count: 0}
  else
    ($issues[0].html_url | split("/issues/")[0]) as $repo_url |
    ($repo_url + "/issues?q=" + ("is:issue is:open label:ready" | @uri)) as $all_url |
    # Display familiar groups first, custom groups next, and unspecified last.
    ($issues | group_by(skill) | sort_by(
      .[0] | skill | if . == "good first issue" then "0"
      elif . == "help wanted" then "1"
      elif . == "unspecified" then "3" else "2" + . end
    )) as $groups |
    {
      count: ($issues | length),
      # Slack uses text as a fallback for notifications and accessibility.
      text: "Issues ready to pick up — \($issues | length) open · <\($all_url)|see all on GitHub>",
      blocks: ([
        {type: "header", text: {type: "plain_text", text: "📋 Issues ready to pick up"}},
        {type: "context", elements: [{type: "mrkdwn",
          text: "\($issues | length) open issue(s) · generated \(now | strftime("%Y-%m-%d %H:%M")) UTC · <\($all_url)|see all on GitHub>"}]}
      ] + [
        # Two blocks per group plus header/context and overflow stay below 50 blocks.
        $groups[:22][] |
        (sort_by(.created_at) | reverse) as $bucket |
        ($bucket[0] | skill) as $skill |
        {type: "divider"},
        {type: "section", text: {type: "mrkdwn", text: (
          "*\($skill | emoji) \($skill | shorten(100))* (\($bucket | length))\n" +
          # Show the five newest issues; link to the full list for the remainder.
          ([$bucket[:5][] |
            "• <\(.html_url)|#\(.number)> \(.title | gsub("[\\r\\n]"; " ") | shorten(250)) — opened \(.created_at[:10])"
          ] | join("\n")) +
          (if ($bucket | length) > 5 then
            "\n… and \(($bucket | length) - 5) more → <\($all_url)|see all on GitHub>"
          else "" end)
        )}}
      ] + (if ($groups | length) > 22 then [
        {type: "section", text: {type: "mrkdwn",
          text: "Additional skill groups → <\($all_url)|see all on GitHub>"}}
      ] else [] end))
    }
  end
' > "$1"
