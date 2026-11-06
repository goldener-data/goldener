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
reachable from the previous stable tag. It fetches merged PRs in batches with
`gh pr list`,
keeps PRs whose merge commit belongs to that set, and deduplicates them. GitHub
bot accounts (including Apps such as Renovate) and branches starting with
`release-v` are excluded.

The dry run prints the prompt without calling Claude or writing a draft. With no
PRs, the script writes a no-change message without calling Claude. Otherwise,
PR titles are Slack-escaped before entering the prompt, and Claude is instructed
to preserve those entities. Claude generates only the body with tools and MCP
servers disabled. Bash adds the heading and release link, then jq rejects empty
bodies, Slack mentions (`<!` and
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

- Environment `release-message-draft` with **selected deployment branches and
  tags**: allow the default branch (`main`) and trusted release tags (for example
  `v*`, with tag creation restricted to maintainers). Store the API key only as
  environment secret `RELEASE_MESSAGE_ANTHROPIC_API_KEY`. The workflow maps it to
  `ANTHROPIC_API_KEY` for Claude; do not define the new secret at repository or
  organization scope. Remove any repository-scoped copy of the same API key
  after migrating it. Environment policies, rather than a branch-editable job
  condition alone, prevent other branches from accessing the key.
- Optional repository variable `ANTHROPIC_MODEL` to select a Claude model.
- Environment `release-message` with **required reviewers**. This is what makes
  Slack publication wait for approval; without reviewers, GitHub starts the job
  immediately after drafting.
- Environment secret `SLACK_BOT_TOKEN`, with Slack's `chat:write` scope and the bot
  invited to the destination channel.
- Environment variable `SLACK_CHANNEL`, preferably the channel ID.

The draft job checks out scripts from the default branch while fetching all
release tags, so announcements for older tags use the maintained scripts.
Manual runs must select the default branch; runs dispatched from other refs are
skipped. Manual runs draft without publishing by default:

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

The baseline is the closest previously published stable release whose commit
is an ancestor of the target tag. Distance is the number of commits in the
candidate-to-target range; ties prefer the most recently published candidate.
Releases on unrelated maintenance branches are ignored. With no eligible ancestor,
the target tag’s entire commit history is included.

Collection uses full commit ancestry, so merge timestamp differences and unrelated
branch merges do not change release membership. It supports release branches and
nested PRs. Merged PRs are fetched through GitHub CLI’s paginated repository listing
without a per-commit request or GitHub search’s 1,000-result limit. Collection cost
now depends on the repository’s total merged PR count; very large repositories can
still hit GitHub rate limits or the draft job timeout.
