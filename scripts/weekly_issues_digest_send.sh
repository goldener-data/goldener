#!/usr/bin/env bash
# Publish a previously generated digest to Slack. Empty digests are skipped.
# Usage: SLACK_BOT_TOKEN=... bash scripts/weekly_issues_digest_send.sh <digest.json>
# Requires Bash, jq, curl. See weekly_issues_digest.md for configuration.
# Stop immediately if validation, request creation, or delivery fails.
set -euo pipefail

if [[ $# != 1 ]]; then
    echo 'Usage: bash scripts/weekly_issues_digest_send.sh <digest.json>' >&2
    exit 1
fi

# Validate the issue count before checking credentials or contacting Slack.
jq -e '(.count | type) == "number" and .count >= 0' "$1" > /dev/null
if jq -e '.count == 0' "$1" > /dev/null; then
    echo 'No ready issues; nothing sent.'
    exit 0
fi
# Empty digests need no credentials; nonempty digests require a bot token.
: "${SLACK_BOT_TOKEN:?Set SLACK_BOT_TOKEN}"
channel=${SLACK_CHANNEL:-'#issues'}
# Keep request/response files private and remove them on success or failure.
send_tmp=$(mktemp -d)
trap 'rm -rf "$send_tmp"' EXIT

# Reuse the generated content, remove its internal count, and add delivery options.
# Disable link previews to keep the digest compact.
jq --arg channel "$channel" '
    if (.text | type) != "string" or (.blocks | type) != "array" then
        error("Expected a generated digest with text and blocks")
    else del(.count) + {channel: $channel, unfurl_links: false, unfurl_media: false} end
' "$1" > "$send_tmp/request.json"
# Do not retry: Slack may have accepted the message before a connection failure.
curl --fail-with-body --silent --show-error --connect-timeout 10 --max-time 60 \
    https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    -H 'Content-Type: application/json; charset=utf-8' \
    --data-binary "@$send_tmp/request.json" > "$send_tmp/response.json"
# Slack can return HTTP 200 with ok=false, so HTTP success alone is insufficient.
jq -e 'if .ok == true then true else error(.error // "Slack rejected the message") end' \
    "$send_tmp/response.json" > /dev/null
echo 'Weekly issues digest published to Slack.'
