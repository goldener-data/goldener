import urllib.error
from datetime import datetime, timezone
from email.message import Message

import pytest

from scripts import release_merged_prs
from scripts.release_merged_prs import (
    collect,
    fetch_tag_date,
    find_previous_release,
    is_excluded,
    parse_labels,
    select_prs,
    summarize_pr,
)

START = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
END = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_release(tag, published_at, draft=False, prerelease=False):
    return {
        "tag_name": tag,
        "published_at": published_at,
        "draft": draft,
        "prerelease": prerelease,
    }


def make_pr(
    number=1,
    merged_at="2026-06-01T12:00:00Z",
    updated_at="2026-06-02T12:00:00Z",
    login="alice",
    branch="add-feature",
    labels=(),
):
    return {
        "number": number,
        "title": f"Pull request {number}",
        "html_url": f"https://github.com/goldener-data/goldener/pull/{number}",
        "body": "Not kept",
        "merged_at": merged_at,
        "updated_at": updated_at,
        "user": {"login": login} if login else None,
        "head": {"ref": branch},
        "labels": [{"name": label} for label in labels],
    }


class TestFindPreviousRelease:
    def test_skips_drafts_prereleases_and_given_tag(self):
        releases = [
            make_release("v3.0.0", "2026-09-14T13:00:00Z"),
            make_release("v2.1.0", None, draft=True),
            make_release("v2.0.0-rc1", "2026-08-01T12:00:00Z", prerelease=True),
            make_release("v2.0.0", "2026-07-01T12:00:00Z"),
            make_release("v1.0.0", "2026-01-01T12:00:00Z"),
        ]

        previous = find_previous_release(releases, "v3.0.0", END)

        assert previous is not None
        assert previous["tag_name"] == "v2.0.0"
        assert find_previous_release(releases[:1], "v3.0.0", END) is None

    def test_uses_dates_for_mixed_tags_and_old_tags(self):
        releases = [
            make_release("v6.5.0", "2026-05-06T12:00:00Z"),
            make_release("6.4.0", "2026-05-04T12:00:00Z"),
            make_release("v6.3.1", "2026-04-15T12:00:00Z"),
        ]
        end_of_old_tag = datetime(2026, 5, 6, 8, 0, tzinfo=timezone.utc)

        previous = find_previous_release(releases, "v6.5.0", end_of_old_tag)

        assert previous is not None
        assert previous["tag_name"] == "6.4.0"


class TestSelectPrs:
    def test_keeps_the_date_range_and_stops_reading_at_start(self):
        def prs():
            yield make_pr(1, merged_at="2026-09-14T12:00:01Z")  # after the end
            yield make_pr(2, merged_at="2026-09-14T12:00:00Z")  # on the end: kept
            yield make_pr(3, merged_at=None)  # closed without a merge
            yield make_pr(4, merged_at="2026-06-01T12:00:00Z")
            yield make_pr(5, merged_at="2026-05-10T12:00:00Z")  # on the start
            yield make_pr(
                6, merged_at="2026-05-01T12:00:00Z", updated_at="2026-05-09T12:00:00Z"
            )
            raise AssertionError("read past the start date")

        selected = select_prs(prs(), START, END)

        assert [pr["number"] for pr in selected] == [4, 2]


class TestIsExcluded:
    def test_covers_bots_release_branches_and_labels(self, monkeypatch):
        assert is_excluded(make_pr(login="github-actions[bot]"))
        assert is_excluded(make_pr(login="dependabot[bot]"))
        assert is_excluded(make_pr(branch="release-v7.0.0"))
        assert not is_excluded(make_pr(login="Copilot"))
        assert parse_labels("") == set()
        assert parse_labels(" skip , other,") == {"skip", "other"}

        monkeypatch.setattr(release_merged_prs, "EXCLUDED_LABELS", set())
        assert not is_excluded(make_pr(labels=["no-post"]))

        monkeypatch.setattr(release_merged_prs, "EXCLUDED_LABELS", {"skip"})
        assert is_excluded(make_pr(labels=["bug", "skip"]))

    def test_release_branch_must_start_with_the_prefix(self):
        assert is_excluded(make_pr(branch="release-v7.0.0"))
        assert not is_excluded(make_pr(branch="fix-release-version"))


class TestSummarizePr:
    def test_keeps_only_five_fields_and_handles_no_pr(self):
        pr = make_pr(7, login="alice", labels=["bug"])

        assert summarize_pr(pr) == {
            "number": 7,
            "title": "Pull request 7",
            "url": "https://github.com/goldener-data/goldener/pull/7",
            "author": "alice",
            "labels": ["bug"],
        }
        assert select_prs([], START, END) == []

    def test_missing_author_is_ghost(self):
        pr = make_pr(login=None)

        assert summarize_pr(pr)["author"] == "ghost"
        assert not is_excluded(pr)


class TestFetchTagDate:
    @pytest.mark.parametrize("code", [404, 422])
    def test_missing_tag_stops_with_a_clear_error(self, monkeypatch, code):
        def fake_request(url, headers):
            raise urllib.error.HTTPError(url, code, "Fake error", Message(), None)

        monkeypatch.setattr(release_merged_prs.digest, "request", fake_request)

        with pytest.raises(RuntimeError, match="Tag v9.9.9 not found"):
            fetch_tag_date("v9.9.9")


class TestCollect:
    def test_connects_the_fetch_functions(self, monkeypatch):
        tag_dates = {"v2.0.0": END, "v1.0.0": START}
        releases = [
            make_release("v2.0.0", "2026-09-14T13:00:00Z"),
            make_release("v1.0.0", "2026-05-10T13:00:00Z"),
        ]
        prs = [make_pr(8), make_pr(9, login="dependabot[bot]")]
        monkeypatch.setattr(
            release_merged_prs, "fetch_tag_date", lambda tag: tag_dates[tag]
        )
        monkeypatch.setattr(release_merged_prs, "fetch_releases", lambda: releases)
        monkeypatch.setattr(release_merged_prs, "iter_closed_prs", lambda: iter(prs))

        assert collect("v2.0.0") == {
            "tag": "v2.0.0",
            "previous_tag": "v1.0.0",
            "start": "2026-05-10T12:00:00+00:00",
            "end": "2026-09-14T12:00:00+00:00",
            "pull_requests": [summarize_pr(prs[0])],
        }
