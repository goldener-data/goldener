#!/usr/bin/env bash
# Usage: REPO=owner/repo bash scripts/release_message_collect.sh <tag> <changes.json>
# Requires a full local clone with release tags, gh, and jq.
set -euo pipefail

if [[ $# != 2 ]]; then
    echo 'Usage: bash scripts/release_message_collect.sh <tag> <changes.json>' >&2
    exit 1
fi

tag=$1
repo=${REPO:-goldener-data/goldener}
collect_tmp=$(mktemp -d)
trap 'rm -rf "$collect_tmp"' EXIT

gh release view "$tag" --repo "$repo" \
    --json tagName,publishedAt,isDraft > "$collect_tmp/release.json"
jq -e '.isDraft == false and .publishedAt != null' "$collect_tmp/release.json" > /dev/null
published=$(jq -r '.publishedAt' "$collect_tmp/release.json")
previous=$(gh api --paginate "repos/$repo/releases?per_page=100" | jq -sr \
    --arg tag "$tag" --arg published "$published" '
    [add[] | select(.draft == false and .prerelease == false and
                   .tag_name != $tag and .published_at < $published)] |
    sort_by(.published_at) | last | .tag_name // ""
')
end=$(git rev-parse --verify "refs/tags/$tag^{commit}")
start=""
if [[ -n "$previous" ]]; then
    start=$(git rev-parse --verify "refs/tags/$previous^{commit}")
    git rev-list "$start..$end" > "$collect_tmp/commits.txt"
else
    git rev-list "$end" > "$collect_tmp/commits.txt"
fi

# Ask for PRs associated with actual release commits, without a search index or
# timestamp window. A PR can be associated with several commits; deduplicate below.
: > "$collect_tmp/prs.jsonl"
while IFS= read -r sha; do
    gh api --paginate "repos/$repo/commits/$sha/pulls?per_page=100" \
        >> "$collect_tmp/prs.jsonl"
done < "$collect_tmp/commits.txt"

jq -sn --arg tag "$tag" --arg previous "$previous" --arg repo "$repo" \
    --arg start "$start" --arg end "$end" \
    --rawfile commits "$collect_tmp/commits.txt" \
    --slurpfile pages "$collect_tmp/prs.jsonl" '
    ($commits | split("\n") | map(select(length > 0)) |
        map({key: ., value: true}) | from_entries) as $included |
    {
        tag: $tag,
        previous_tag: (if $previous == "" then null else $previous end),
        start: (if $start == "" then null else $start end),
        end: $end,
        pull_requests: [($pages | add // [])[] |
            select(.merged_at != null and .base.repo.full_name == $repo) |
            select($included[.merge_commit_sha] == true) |
            select(.user.login != "dependabot[bot]" and
                   .user.login != "github-actions[bot]") |
            select(.head.ref | startswith("release-v") | not)
        ] | unique_by(.number) | sort_by(.merged_at, .number) |
            map({number, title, url: .html_url, author: (.user.login // "ghost"),
                 labels: [.labels[].name]})
    }
' > "$2"
