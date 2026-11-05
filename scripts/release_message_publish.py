"""Post the message of a release to Slack.

The message is the MESSAGE variable if it is set, otherwise the draft written by
scripts.release_message_draft. Without SEND_TO_SLACK=True it is only printed.

Run it from the root of the repository, so that the other scripts can be imported:
  python -m scripts.release_message_publish v7.0.0 draft_dir

`draft_dir` holds changes.json and draft.txt. It can be missing when MESSAGE is set.
Errors go to stderr with exit code 1.

Environment variables:
  MESSAGE          edited text, which replaces the draft when it is not empty
  SEND_TO_SLACK    "True" to post; anything else only prints the message
  SLACK_BOT_TOKEN  required to post
  SLACK_CHANNEL    required to post, no default
"""

import json
import os
import sys
import urllib.error
from pathlib import Path

import scripts.weekly_issues_digest as digest
from scripts.release_message_draft import (
    FORBIDDEN_MARKERS,
    MAX_MESSAGE_LENGTH,
    read_required_env,
)

SLACK_URL = "https://slack.com/api/chat.postMessage"


def read_draft(tag: str, draft_dir: Path) -> str:
    """Text of the draft, after checking that it announces this release."""
    changes_file = draft_dir / "changes.json"
    changes = json.loads(changes_file.read_text(encoding="utf-8"))
    if changes["tag"] != tag:
        raise RuntimeError(f"The draft is for {changes['tag']}, not for {tag}.")
    if not changes["pull_requests"]:
        raise RuntimeError(f"No pull request in {tag}: nothing to announce.")
    return (draft_dir / "draft.txt").read_text(encoding="utf-8")


def load_message(tag: str, draft_dir: Path) -> str:
    """The edited message if there is one, otherwise the draft."""
    message = os.environ.get("MESSAGE", "")
    text = message if message else read_draft(tag, draft_dir)
    return text.strip()


def check_message(text: str) -> None:
    """Stop on a message that is empty, too long or that would ping people."""
    if not text:
        raise RuntimeError("The message is empty.")
    if len(text) > MAX_MESSAGE_LENGTH:
        raise RuntimeError(
            f"The message has {len(text)} characters, "
            f"more than the {MAX_MESSAGE_LENGTH} allowed."
        )
    for marker in FORBIDDEN_MARKERS:
        if marker in text:
            raise RuntimeError(f"The message holds a Slack mention: {marker}")


def send_to_slack(token: str, channel: str, text: str) -> None:
    """One POST to Slack. Slack responds 200 even on error, so `ok` is checked."""
    payload = {
        "channel": channel,
        "text": text,
        "unfurl_links": False,
        "unfurl_media": False,
    }
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {token}",
    }
    # No retries: if Slack accepted the message but the response was lost,
    # a retry would post it twice.
    data = json.dumps(payload).encode()
    result, _ = digest.request(SLACK_URL, headers, data, retries=0)
    if not result.get("ok"):
        raise RuntimeError(f"Slack rejected the message: {result.get('error')}")


# ---------- entry point ----------


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m scripts.release_message_publish <tag> <draft_dir>")
    text = load_message(sys.argv[1], Path(sys.argv[2]))
    check_message(text)
    if os.environ.get("SEND_TO_SLACK") != "True":
        channel = os.environ.get("SLACK_CHANNEL") or "(not set)"
        print(f"Channel: {channel}\n\n{text}\n\nTEST: nothing sent.")
        return
    token = read_required_env("SLACK_BOT_TOKEN")
    channel = read_required_env("SLACK_CHANNEL")
    send_to_slack(token, channel, text)
    print(f"Sent to {channel}.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP error {exc.code} on {exc.url} — {exc.reason}")
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(f"Network error: {exc}")
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        sys.exit(f"Error: {exc}")
