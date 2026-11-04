"""Pull requests merged since the previous release of the repository.

Run it from the root of the repository, so that the digest script can be imported:
  python -m scripts.release_merged_prs v7.0.0

The result is printed as JSON on stdout, and errors go to stderr with exit code 1.

Environment variables:
  EXCLUDED_LABELS  comma-separated label names to leave out, empty by default
  REPO             default goldener-data/goldener
  GITHUB_TOKEN     avoids the anonymous rate limit (60/h)
"""

import json
import os
import sys
import urllib.error
import urllib.parse
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any, TypedDict

import scripts.weekly_issues_digest as digest

EXCLUDED_BOTS = {"github-actions[bot]", "dependabot[bot]"}
RELEASE_BRANCH_PREFIX = "release-v"
GHOST_LOGIN = "ghost"


def parse_labels(value: str) -> set[str]:
    """Label names from a comma-separated string. An empty string gives no label."""
    return {label.strip() for label in value.split(",") if label.strip()}


EXCLUDED_LABELS = parse_labels(os.environ.get("EXCLUDED_LABELS", ""))


class PullRequestSummary(TypedDict):
    number: int
    # Written by the author of the pull request: this is untrusted text.
    title: str
    url: str
    author: str
    labels: list[str]


class ReleaseChanges(TypedDict):
    tag: str
    previous_tag: str | None
    start: str | None
    end: str
    pull_requests: list[PullRequestSummary]


def parse_date(value: str) -> datetime:
    """Date of a GitHub field, such as 2026-09-14T19:45:45Z."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def author_login(pr: dict[str, Any]) -> str:
    """Login of the author, or "ghost" when GitHub gives no user (deleted account)."""
    user = pr["user"]
    return user["login"] if user else GHOST_LOGIN


def find_previous_release(
    releases: Iterable[dict[str, Any]], tag: str, end: datetime
) -> dict[str, Any] | None:
    """Latest published release before `end`, or None for a first release.

    Dates are compared instead of versions because the tags do not all have the
    same format (6.4.0 has no leading v).
    """
    candidates = [
        release
        for release in releases
        if not release["draft"]
        and not release["prerelease"]
        and release["tag_name"] != tag
        and parse_date(release["published_at"]) < end
    ]
    return max(
        candidates,
        key=lambda release: parse_date(release["published_at"]),
        default=None,
    )


def is_excluded(pr: dict[str, Any]) -> bool:
    """True for pull requests from a listed bot, a release branch or an excluded label."""
    if author_login(pr) in EXCLUDED_BOTS:
        return True
    if pr["head"]["ref"].startswith(RELEASE_BRANCH_PREFIX):
        return True
    return any(label["name"] in EXCLUDED_LABELS for label in pr["labels"])


def is_merged_between(
    pr: dict[str, Any], start: datetime | None, end: datetime
) -> bool:
    """True if merged after `start` (None means no limit) and up to `end` included."""
    if pr["merged_at"] is None:
        return False
    merged_at = parse_date(pr["merged_at"])
    return (start is None or merged_at > start) and merged_at <= end


def select_prs(
    prs: Iterable[dict[str, Any]], start: datetime | None, end: datetime
) -> list[dict[str, Any]]:
    """Pull requests to keep, oldest merge first.

    `prs` must be sorted by update date, newest first. A pull request cannot be
    updated before its merge, so reading stops at the first one older than `start`.
    """
    selected = []
    for pr in prs:
        if start is not None and parse_date(pr["updated_at"]) < start:
            break
        if is_merged_between(pr, start, end) and not is_excluded(pr):
            selected.append(pr)
    return sorted(selected, key=lambda pr: parse_date(pr["merged_at"]))


def summarize_pr(pr: dict[str, Any]) -> PullRequestSummary:
    """The fields kept for a pull request. The body is left out on purpose."""
    return {
        "number": pr["number"],
        "title": pr["title"],
        "url": pr["html_url"],
        "author": author_login(pr),
        "labels": [label["name"] for label in pr["labels"]],
    }


# ---------- network ----------


def fetch_releases() -> list[dict[str, Any]]:
    """All the releases of the repository."""
    return list(
        digest.paginate(f"{digest.API}/repos/{digest.REPO}/releases?per_page=100")
    )


def fetch_tag_date(tag: str) -> datetime:
    """Date of the commit the tag points at. A missing tag stops with a clear error."""
    ref = urllib.parse.quote(tag, safe="")
    url = f"{digest.API}/repos/{digest.REPO}/commits/{ref}"
    try:
        commit, _ = digest.request(url, digest.github_headers())
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 422):
            raise RuntimeError(f"Tag {tag} not found in {digest.REPO}.") from exc
        raise
    return parse_date(commit["commit"]["committer"]["date"])


def iter_closed_prs() -> Iterator[dict[str, Any]]:
    """Closed pull requests into main, most recently updated first, page by page."""
    params = urllib.parse.urlencode(
        {
            "state": "closed",
            "base": "main",
            "sort": "updated",
            "direction": "desc",
            "per_page": 100,
        }
    )
    return digest.paginate(f"{digest.API}/repos/{digest.REPO}/pulls?{params}")


# ---------- entry point ----------


def collect(tag: str) -> ReleaseChanges:
    """The pull requests merged between the previous release and `tag`."""
    end = fetch_tag_date(tag)
    previous = find_previous_release(fetch_releases(), tag, end)
    previous_tag = previous["tag_name"] if previous else None
    start = fetch_tag_date(previous_tag) if previous_tag else None
    prs = select_prs(iter_closed_prs(), start, end)
    return {
        "tag": tag,
        "previous_tag": previous_tag,
        "start": start.isoformat() if start else None,
        "end": end.isoformat(),
        "pull_requests": [summarize_pr(pr) for pr in prs],
    }


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m scripts.release_merged_prs <tag>")
    print(json.dumps(collect(sys.argv[1]), indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP error {exc.code} on {exc.url} — {exc.reason}")
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(f"Network error: {exc}")
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")
