# Draft a release message with GitHub CLI and Claude Code

Requires Bash, `gh`, `jq`, and an authenticated `claude` CLI. Claude can use its
existing login or `ANTHROPIC_API_KEY`; `ANTHROPIC_MODEL` optionally selects a model.

From a full clone of the repository with its release tags available, collect the
PRs included in a published release (the previous stable release is selected
automatically):

```bash
git fetch --tags origin
bash scripts/release_message_collect.sh v7.0.0 changes.json
DRY_RUN=True bash scripts/release_message_draft.sh changes.json draft.txt
bash scripts/release_message_draft.sh changes.json draft.txt
```

The collector walks commits reachable from the target tag, excluding commits
reachable from the previous stable tag. It requests associated PRs from GitHub,
keeps merged PRs whose merge commit belongs to that set, and deduplicates them.
Bot PRs and branches starting with `release-v` are excluded.

The dry run prints the prompt without calling Claude or writing a draft. With no
PRs, the script writes a no-change message without calling Claude. Otherwise,
Claude generates only the body with tools and MCP servers disabled. Bash adds the
heading and release link, then jq rejects empty bodies, Slack mentions (`<!` and
`<@`), unsuccessful CLI responses, and messages over 3,000 characters before
writing the output. Review the resulting text before publication.

For another repository, use a full clone of that repository and set
`REPO=owner/repo` when running both scripts.

## GitHub Actions and Slack

The `Draft and publish the release message` workflow starts on `release.published`,
including releases published from a saved draft. It collects PRs using `gh`, drafts
with Claude CLI, and shows the message in the run summary. It then passes the saved
draft to the Slack publication job. No PRs means no Claude call and no Slack post.

Configure the repository before enabling publication:

- Repository secret `ANTHROPIC_API_KEY` for the draft job.
- Optional repository variable `ANTHROPIC_MODEL` to select a Claude model.
- Environment `release-message` with **required reviewers**. This is what makes
  Slack publication wait for approval; without reviewers, GitHub starts the job
  immediately after drafting.
- Environment secret `SLACK_BOT_TOKEN`, with Slack's `chat:write` scope and the bot
  invited to the destination channel.
- Environment variable `SLACK_CHANNEL`, preferably the channel ID.

Manual runs draft without publishing by default:

```bash
gh workflow run release-message.yaml -f tag=v7.0.0
gh workflow run release-message.yaml -f tag=v7.0.0 -f dry_run=true
gh workflow run release-message.yaml -f tag=v7.0.0 -f publish=true
```

Before approving the publication job, read the draft in the run summary. The
workflow downloads the same artifact for publication; it does not generate the
message again. Slack publication uses `curl` and checks both HTTP failures and
Slack's JSON `ok` field. It never retries a post automatically. Before rerunning a
failed publication, check whether the message already reached Slack: rerunning can
post a duplicate. Concurrency serializes runs for the same tag but does not deduplicate them.

The previous stable release is selected by publication date. First releases
include the target tag’s entire commit history. Collection uses commit ancestry,
so merge timestamp differences and unrelated branch merges do not change release
membership. It supports release branches and paginates the associated-PR API
without GitHub search’s 1,000-result limit. Each commit requires an API request;
large release histories can hit GitHub rate limits or the draft job timeout.
