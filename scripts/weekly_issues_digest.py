#!/usr/bin/env python3
"""Weekly digest of GitHub issues ready for development, posted to Slack.

Usage:
  python weekly_issues_digest.py                 # TEST: prints, sends nothing
  python weekly_issues_digest.py --list-labels   # lists the repo's real labels
  SEND_TO_SLACK=1 SLACK_BOT_TOKEN=xoxb-... python weekly_issues_digest.py   # real send

Environment variables (all optional except the Slack token in live mode):
  REPO             default goldener-data/goldener
  READY_LABEL      default ready          (EXACT label name, see --list-labels)
  SKILL_REGEX      default skill|level|good first issue|help wanted
  GITHUB_TOKEN     avoids the anonymous rate limit (60/h)
  SEND_TO_SLACK    "1" to actually send
  SLACK_BOT_TOKEN  bot token (xoxb-...)
  SLACK_CHANNEL    default #issues

Note: "priority" is NOT handled here. On goldener-data/goldener, priority is a
GitHub Projects (v2) field, not a label -- so it isn't readable through the
issues REST API this script uses. Only the "ready" label (filter) and
skill-type labels (good first issue / help wanted) are used.

This script lists ALL open issues carrying the "ready" label on every run.
Since most issues in the repo carry this label (it isn't a "just added"
marker), each issue's opening date is shown and the list is sorted newest
first: readers see new issues at a glance, while older ones never disappear
from the list.

No external dependencies: standard library only.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timezone
from email.message import Message
from typing import Any

API = "https://api.github.com"
REPO = os.environ.get("REPO", "goldener-data/goldener")
READY_LABEL = os.environ.get("READY_LABEL", "ready")
SKILL_RE = re.compile(
    os.environ.get("SKILL_REGEX", r"skill|level|good first issue|help wanted"), re.I
)
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "#issues")
CHUNK_LIMIT = 3000  # characters per Slack message, to stay readable

UNKNOWN = "unspecified"


# ---------- network ----------


def request(
    url: str,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    retries: int = 3,
) -> tuple[Any, Message]:
    """JSON HTTP call with retries (429 / 5xx / network errors)."""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers or {}, data=data)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.load(resp), resp.headers
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After")
            retriable = exc.code in (429, 500, 502, 503, 504) or (
                exc.code == 403 and retry_after
            )
            if not retriable or attempt == retries:
                raise
            time.sleep(min(int(retry_after) if retry_after else 2**attempt, 30))
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")  # for mypy: loop always returns or raises


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "weekly-issues-digest",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def paginate(url: str | None) -> Iterator[dict[str, Any]]:
    """Follows GitHub's Link: rel="next" header, page after page."""
    headers = github_headers()
    while url:
        items, resp_headers = request(url, headers)
        yield from items
        match = re.search(r'<([^>]+)>;\s*rel="next"', resp_headers.get("Link", ""))
        url = match.group(1) if match else None


# ---------- business logic ----------


def fetch_ready_issues() -> list[dict[str, Any]]:
    """All open issues carrying the 'ready' label, regardless of their date."""
    params = urllib.parse.urlencode(
        {
            "state": "open",
            "labels": READY_LABEL,
            "sort": "created",
            "direction": "desc",
            "per_page": 100,
        }
    )
    issues = []
    for issue in paginate(f"{API}/repos/{REPO}/issues?{params}"):
        if "pull_request" in issue:  # the API mixes issues and pull requests
            continue
        issues.append(issue)
    return issues


def strip_decor(text: str) -> str:
    """Removes emojis, punctuation, and leading/trailing whitespace."""
    return re.sub(r"^[\W_]+|[\W_]+$", "", text)


def label_value(labels: list[dict[str, Any]], regex: re.Pattern[str]) -> str:
    """Value of the first matching label. 'good first issue' -> 'good first issue'."""
    for label in labels:
        name = label["name"]
        if regex.search(name):
            value = strip_decor(regex.sub("", name, count=1))
            return value or strip_decor(name)
    return UNKNOWN


def slack_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_entries(issues: list[dict[str, Any]]) -> list[str]:
    """One line per issue, sorted by opening date descending (newest first)."""
    entries = []
    for issue in issues:
        skill = label_value(issue["labels"], SKILL_RE)
        created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
        line = (
            f"• <{issue['html_url']}|#{issue['number']}> {slack_escape(issue['title'])}"
            f" — Opened {created:%Y-%m-%d} · Skill: {skill}"
        )
        entries.append((created, line))
    entries.sort(key=lambda e: e[0], reverse=True)  # newest first
    return [e[1] for e in entries]


def build_messages(lines: list[str], now: datetime) -> list[str]:
    """A single digest, split into several messages only if it's very long."""
    header = f"*Issues ready to pick up* — {len(lines)} open as of {now:%Y-%m-%d}"
    messages, current = [], header
    for line in lines:
        if len(current) + len(line) + 1 > CHUNK_LIMIT:
            messages.append(current)
            current = line
        else:
            current += "\n" + line
    messages.append(current)
    return messages


def send_to_slack(text: str) -> None:
    payload = json.dumps({"channel": SLACK_CHANNEL, "text": text}).encode()
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
    }
    result, _ = request("https://slack.com/api/chat.postMessage", headers, payload)
    if not result.get("ok"):  # Slack responds 200 even on error
        raise RuntimeError(f"Slack rejected the message: {result.get('error')}")


# ---------- entry point ----------


def list_labels() -> None:
    print(f"Labels for {REPO}:")
    for label in paginate(f"{API}/repos/{REPO}/labels?per_page=100"):
        print(f"  {label['name']}")


def main() -> None:
    if "--list-labels" in sys.argv:
        list_labels()
        return

    live = os.environ.get("SEND_TO_SLACK") == "1"
    if live and not os.environ.get("SLACK_BOT_TOKEN"):
        sys.exit("Error: SEND_TO_SLACK=1 but SLACK_BOT_TOKEN is missing.")

    now = datetime.now(timezone.utc)
    issues = fetch_ready_issues()

    if not issues:
        print(f"No open issue with the '{READY_LABEL}' label right now.")
        print(
            "If that's unexpected, check the exact label name: python weekly_issues_digest.py --list-labels"
        )
        return  # nothing to send: don't spam the channel

    messages = build_messages(build_entries(issues), now)
    for message in messages:
        if live:
            send_to_slack(message)
        else:
            print(message, end="\n\n")
    print(
        f"{len(issues)} issue(s) — {'sent to Slack' if live else 'TEST: nothing sent'}."
    )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP error {exc.code} on {exc.url} — {exc.reason}")
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(f"Network error: {exc}")
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")
