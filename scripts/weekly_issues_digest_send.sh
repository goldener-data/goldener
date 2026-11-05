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
if ! jq -e '(.count | type) == "number" and .count >= 0 and .count == (.count | floor)' "$1" > /dev/null; then
    printf 'Invalid digest %s: expected count to be a nonnegative integer.\n' "$1" >&2
    exit 1
fi
if jq -e '.count == 0' "$1" > /dev/null; then
    echo 'No ready issues; nothing sent.'
    exit 0
fi
# Empty digests need no credentials; nonempty digests require a bot token.
: "${SLACK_BOT_TOKEN:?Set SLACK_BOT_TOKEN}"
channel=${SLACK_CHANNEL:-'#issues'}
# Keep request/response files private and remove them on success or failure.
umask 077
send_tmp=$(mktemp -d)
trap 'rm -rf "$send_tmp"' EXIT

# Reuse the generated content, remove its internal count, and add delivery options.
# Disable link previews to keep the digest compact.
jq --arg channel "$channel" '
    if (.text | type) != "string" or (.blocks | type) != "array" then
        error("Expected a generated digest with text and blocks")
    else del(.count) + {channel: $channel, unfurl_links: false, unfurl_media: false} end
' "$1" > "$send_tmp/request.json"
# Read the token from a private file, keeping it out of curl process arguments.
printf 'Authorization: Bearer %s\n' "$SLACK_BOT_TOKEN" > "$send_tmp/headers"
# Do not retry: Slack may have accepted the message before a connection failure.
curl_status=0
curl --fail-with-body --silent --show-error --connect-timeout 10 --max-time 60 \
    https://slack.com/api/chat.postMessage \
    -H "@$send_tmp/headers" \
    -H 'Content-Type: application/json; charset=utf-8' \
    --data-binary "@$send_tmp/request.json" > "$send_tmp/response.json" || curl_status=$?
# --fail-with-body preserves Slack JSON on HTTP errors. Report its reason before
# exiting, while preserving curl transport/HTTP failures even if no JSON arrived.
if (( curl_status != 0 )); then
    slack_error=$(jq -r '.error // empty' "$send_tmp/response.json" 2>/dev/null || true)
    if [[ -n "$slack_error" ]]; then
        printf 'Slack rejected the request: %s (curl exit %s).\n' "$slack_error" "$curl_status" >&2
    fi
    exit "$curl_status"
fi
# Slack can return HTTP 200 with ok=false, so HTTP success alone is insufficient.
jq -e 'if .ok == true then true else error(.error // "Slack rejected the message") end' \
    "$send_tmp/response.json" > /dev/null
echo 'Weekly issues digest published to Slack.'
