# Weekly issues digest

The workflow `.github/workflows/weekly-issues-digest.yaml` calls two Bash scripts:

- `weekly_issues_digest_generate.sh` fetches ready issues and generates Slack JSON
  using GitHub CLI and its built-in `--jq` formatter. It never posts to Slack.
- `weekly_issues_digest_send.sh` publishes that saved JSON using `curl`, with `jq`
  to build the request and check Slack's response.

The workflow runs on Mondays at 09:17 UTC and supports manual runs from Actions.

## Generate and review locally

Requires Bash and an authenticated GitHub CLI (`gh auth login` or `GH_TOKEN`).
The review command below also requires `jq`.
The GitHub token needs read access to issues in the repository.

```bash
REPO=goldener-data/goldener bash scripts/weekly_issues_digest_generate.sh /tmp/digest.json
jq . /tmp/digest.json
```

The output path is required and its parent directory must exist. `REPO` defaults
to `goldener-data/goldener`. Generation replaces the output file; a failed command
may leave partial output, so only send it after generation succeeds.

The script fetches all open issues with the exact label `ready`, using GitHub
CLI's internal pagination. It groups issues by the first label matching
`skill|level|good first issue|help wanted` (case insensitive); skill names are
normalized to lowercase. Issues without such labels appear under `unspecified`.
Within each group, the newest issues appear first with their creation date and
GitHub link.

Each group shows up to five issues, with a link to all ready issues for overflow.
At most 22 groups are shown, followed by a link if more groups exist. Titles and
group names are escaped and shortened to fit Slack's message limits. The JSON
contains `count`, Slack fallback `text`, and `blocks`. The fallback mirrors the
displayed header, context, skill groups, issue summaries, and overflow links for
notifications and screen readers. With no ready issues it contains only `{"count":0}`.

## Publish the reviewed digest

Requires Bash, `jq`, `curl`, a Slack bot token with `chat:write`, and the bot
invited to the destination channel. Set `SLACK_BOT_TOKEN` in your environment
before running:

```bash
SLACK_CHANNEL='#issues' bash scripts/weekly_issues_digest_send.sh /tmp/digest.json
```

`SLACK_CHANNEL` defaults to `#issues`; a channel ID is also supported. Empty
digests are skipped without requiring a Slack token. Nonempty digests require
`SLACK_BOT_TOKEN`. The sender checks HTTP errors and Slack's JSON `ok` field.
Invalid counts report the input file and expected format; HTTP failures also
report Slack's `error` when present. The Authorization header is passed through a
private temporary file, removed on exit, so curl arguments do not contain the token.
It makes one posting attempt and does not retry automatically. Before rerunning
a failed send, check whether Slack already received it to avoid duplicate posts.

## GitHub Actions configuration

Configure the `slack-digest` environment with:

- Secret `SLACK_BOT_TOKEN` for the Slack bot.
- Optional variable `SLACK_CHANNEL` (defaults to `#issues`).

The workflow's GitHub token has `contents: read` for checkout and `issues: read`
for collection. It saves the generated JSON in the runner's temporary directory,
then sends that same file only when `count` is positive. Triggering this workflow
from GitHub posts the digest; to preview without sending, run only the generator
locally as shown above.
