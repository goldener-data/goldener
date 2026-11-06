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
end=$(git rev-parse --verify "refs/tags/$tag^{commit}")
# Publication dates only determine eligibility. Choose the closest stable release
# on this tag history, so a recent maintenance release on another line is ignored.
gh api --paginate "repos/$repo/releases?per_page=100" | jq -sr \
    --arg tag "$tag" --arg published "$published" '
    [add[] | select(.draft == false and .prerelease == false and
                   .tag_name != $tag and .published_at < $published)] |
    sort_by(.published_at) | reverse | .[].tag_name
' > "$collect_tmp/candidates.txt"
previous=""
best_distance=""
while IFS= read -r candidate; do
    candidate_sha=$(git rev-parse --verify "refs/tags/$candidate^{commit}")
    if git merge-base --is-ancestor "$candidate_sha" "$end"; then
        distance=$(git rev-list --count "$candidate_sha..$end")
        if [[ -z "$best_distance" ]] || (( distance < best_distance )); then
            previous=$candidate
            best_distance=$distance
        fi
    else
        status=$?
        if (( status != 1 )); then exit "$status"; fi
    fi
done < "$collect_tmp/candidates.txt"
start=""
if [[ -n "$previous" ]]; then
    start=$(git rev-parse --verify "refs/tags/$previous^{commit}")
    git rev-list "$start..$end" > "$collect_tmp/commits.txt"
else
    git rev-list "$end" > "$collect_tmp/commits.txt"
fi

# Fetch merged PRs in batches through the repository connection, rather than one
# request per commit. gh paginates internally; the large limit avoids its default
# 30-PR cap. Matching against the full commit set preserves nested merge history.
gh pr list --repo "$repo" --state merged --limit 2147483647 \
    --json number,title,url,author,labels,headRefName,mergedAt,mergeCommit \
    > "$collect_tmp/prs.json"

jq -sn --arg tag "$tag" --arg previous "$previous" \
    --arg start "$start" --arg end "$end" \
    --rawfile commits "$collect_tmp/commits.txt" \
    --slurpfile prs "$collect_tmp/prs.json" '
    ($commits | split("\n") | map(select(length > 0)) |
        map({key: ., value: true}) | from_entries) as $included |
    {
        tag: $tag,
        previous_tag: (if $previous == "" then null else $previous end),
        start: (if $start == "" then null else $start end),
        end: $end,
        pull_requests: [$prs[0][] |
            select($included[.mergeCommit.oid // ""] == true) |
            select(.author.is_bot != true) |
            select(.headRefName | startswith("release-v") | not)
        ] | unique_by(.number) | sort_by(.mergedAt, .number) |
            map({number, title, url, author: (.author.login // "ghost"),
                 labels: [.labels[].name]})
    }
' > "$2"
