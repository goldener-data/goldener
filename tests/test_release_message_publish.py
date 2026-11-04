import json

import pytest

from scripts.release_message_publish import (
    check_message,
    load_message,
    main,
    read_draft,
)

FAKE_TOKEN = "fake-token"
FAKE_CHANNEL = "#fake-channel"
DRAFT = "*Goldener v2.0.0 is out!*\n\nThe body\n"
REQUEST = "scripts.weekly_issues_digest.request"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("MESSAGE", "SEND_TO_SLACK", "SLACK_BOT_TOKEN", "SLACK_CHANNEL"):
        monkeypatch.delenv(name, raising=False)


def write_draft(draft_dir, tag="v2.0.0", pull_requests=({"number": 1},)):
    changes = {"tag": tag, "pull_requests": list(pull_requests)}
    (draft_dir / "changes.json").write_text(json.dumps(changes), encoding="utf-8")
    (draft_dir / "draft.txt").write_text(DRAFT, encoding="utf-8")


def run_main(monkeypatch, draft_dir):
    monkeypatch.setattr("sys.argv", ["publish", "v2.0.0", str(draft_dir)])
    main()


def fail_if_called(*args, **kwargs):
    raise AssertionError("must not be called")


class TestReadDraft:
    def test_returns_the_draft_of_the_release(self, tmp_path):
        write_draft(tmp_path)

        assert read_draft("v2.0.0", tmp_path) == DRAFT

    def test_stops_on_a_tag_mismatch(self, tmp_path):
        write_draft(tmp_path, tag="v1.0.0")

        with pytest.raises(RuntimeError, match="for v1.0.0, not for v2.0.0"):
            read_draft("v2.0.0", tmp_path)

    def test_stops_without_pull_requests(self, tmp_path):
        write_draft(tmp_path, pull_requests=())

        with pytest.raises(RuntimeError, match="nothing to announce"):
            read_draft("v2.0.0", tmp_path)


class TestLoadMessage:
    def test_message_replaces_the_draft_without_a_draft_dir(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("MESSAGE", "Edited text")

        assert load_message("v2.0.0", tmp_path / "missing") == "Edited text"

    def test_message_replaces_a_draft_of_another_release(self, monkeypatch, tmp_path):
        write_draft(tmp_path, tag="v1.0.0")
        monkeypatch.setenv("MESSAGE", "Edited text")

        assert load_message("v2.0.0", tmp_path) == "Edited text"

    def test_empty_message_uses_the_draft(self, monkeypatch, tmp_path):
        write_draft(tmp_path)
        monkeypatch.setenv("MESSAGE", "")

        assert load_message("v2.0.0", tmp_path) == DRAFT.strip()


class TestCheckMessage:
    def test_accepts_a_message_at_the_limit(self):
        check_message("x" * 3000)

    def test_rejects_a_too_long_message(self):
        with pytest.raises(RuntimeError, match="more than the 3000 allowed"):
            check_message("x" * 3001)

    def test_rejects_an_empty_message(self):
        with pytest.raises(RuntimeError, match="empty"):
            check_message("")

    @pytest.mark.parametrize("text", ["Hello <!channel>", "Hello <@U123>"])
    def test_rejects_a_mention(self, text):
        with pytest.raises(RuntimeError, match="Slack mention"):
            check_message(text)


class TestMain:
    def test_blank_message_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MESSAGE", "  \n")
        monkeypatch.setattr(REQUEST, fail_if_called)

        with pytest.raises(RuntimeError, match="empty"):
            run_main(monkeypatch, tmp_path)

    @pytest.mark.parametrize("value", [None, "False", "true"])
    def test_only_prints_unless_send_to_slack_is_true(
        self, monkeypatch, capsys, tmp_path, value
    ):
        write_draft(tmp_path)
        if value is not None:
            monkeypatch.setenv("SEND_TO_SLACK", value)
        monkeypatch.setenv("SLACK_CHANNEL", FAKE_CHANNEL)
        monkeypatch.setattr(REQUEST, fail_if_called)

        run_main(monkeypatch, tmp_path)

        output = capsys.readouterr().out
        assert DRAFT.strip() in output
        assert FAKE_CHANNEL in output

    def test_print_mode_needs_no_channel(self, monkeypatch, capsys, tmp_path):
        write_draft(tmp_path)
        monkeypatch.setattr(REQUEST, fail_if_called)

        run_main(monkeypatch, tmp_path)

        assert "(not set)" in capsys.readouterr().out

    def test_sends_exactly_one_request(self, monkeypatch, capsys, tmp_path):
        write_draft(tmp_path)
        monkeypatch.setenv("SEND_TO_SLACK", "True")
        monkeypatch.setenv("SLACK_BOT_TOKEN", FAKE_TOKEN)
        monkeypatch.setenv("SLACK_CHANNEL", FAKE_CHANNEL)
        calls = []

        def fake_request(url, headers, data, retries):
            calls.append((url, headers, json.loads(data), retries))
            return {"ok": True}, {}

        monkeypatch.setattr(REQUEST, fake_request)

        run_main(monkeypatch, tmp_path)

        assert len(calls) == 1
        url, headers, payload, retries = calls[0]
        assert url == "https://slack.com/api/chat.postMessage"
        assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        assert payload == {
            "channel": FAKE_CHANNEL,
            "text": DRAFT.strip(),
            "unfurl_links": False,
            "unfurl_media": False,
        }
        assert retries == 0
        assert FAKE_TOKEN not in capsys.readouterr().out

    def test_stops_when_slack_answers_not_ok(self, monkeypatch, tmp_path):
        write_draft(tmp_path)
        monkeypatch.setenv("SEND_TO_SLACK", "True")
        monkeypatch.setenv("SLACK_BOT_TOKEN", FAKE_TOKEN)
        monkeypatch.setenv("SLACK_CHANNEL", FAKE_CHANNEL)
        monkeypatch.setattr(
            REQUEST, lambda *args, **kwargs: ({"ok": False, "error": "no_access"}, {})
        )

        with pytest.raises(RuntimeError, match="rejected the message: no_access"):
            run_main(monkeypatch, tmp_path)

    @pytest.mark.parametrize(
        ("missing", "present", "value"),
        [
            ("SLACK_BOT_TOKEN", "SLACK_CHANNEL", FAKE_CHANNEL),
            ("SLACK_CHANNEL", "SLACK_BOT_TOKEN", FAKE_TOKEN),
        ],
    )
    def test_missing_variable_stops_without_printing_a_value(
        self, monkeypatch, capsys, tmp_path, missing, present, value
    ):
        write_draft(tmp_path)
        monkeypatch.setenv("SEND_TO_SLACK", "True")
        monkeypatch.setenv(present, value)
        monkeypatch.setattr(REQUEST, fail_if_called)

        with pytest.raises(RuntimeError, match=missing) as error:
            run_main(monkeypatch, tmp_path)

        assert value not in str(error.value)
        assert value not in capsys.readouterr().out
