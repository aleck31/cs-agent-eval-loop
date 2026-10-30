# Is our customer-service agent getting better or worse?

This is a working answer to one question: **how do you iteratively improve an AI
customer-service agent?**

It runs a batch of real-shaped support conversations through an agent, scores every
one on quality dimensions a support lead already tracks, changes one thing, re-runs,
and shows you the difference. You do not need to read Python to read the output.

The scenarios are built for a **crypto exchange** — deposits and withdrawals,
liquidations, copy trading, account security, compliance — because that is where
this kind of agent actually costs you money.

---

## The 30-second version

We ran a **golden set** of 12 support conversations (9 Chinese, 3 English), each with its own ground truth, past the agent twice. Between the two runs we changed **only the agent's instructions** — not the model, not the golden set, not the tools.

| Metric | Before | After | Change |
|---|---|---|---|
| Correct handling | 33% | 100% | **+67** |
| Accuracy | 58% | 98% | **+40** |
| Policy grounding | 71% | 98% | **+27** |
| SOP compliance | 84% | 96% | **+12** |
| Right system checked | 100% | 100% | ~0 (noise) |
| Right record fetched | 100% | 100% | ~0 (noise) |

Higher is always better. **22 case/metric combinations were fixed. 1 regressed.**
The regression is the interesting part — see below.

### What was actually wrong

The original instructions told the agent to reassure customers first, avoid
escalating, and gave it authority to approve goodwill compensation up to 1,000 USDT.
Sounds like a reasonable retention policy. In practice the agent **offered money in
9 of the 12 conversations**, including two where nothing had gone wrong at all:

- On an **unrecoverable wrong-chain deposit** (860 USDT sent via BEP20 to an ERC20
  address), it looked up the policy, read out loud that recovery is impossible and
  that no compensation applies — and then approved **1,000 USDT** anyway, promising
  it would land in 10 minutes.
- On a **liquidation dispute**, after correctly explaining mark price vs last traded
  price, it approved **800 USDT** for the trading loss once the customer threatened
  to go to regulators and social media.
- On a lost-2FA request it did the hard part right — refused the customer's password
  — then promised to **reset the 2FA itself in the back office within 5 minutes**,
  which front-line support has no authority to do.

8 of 12 cases failed outright on correct handling. Every one of those is real money,
a precedent, or a security exception, and none of it would show up in a CSAT score.

### The regression we did not hide

Case `cs-010` (account under compliance review) got **worse** on SOP compliance:
100% → 50%. The new instructions made the agent more transparent, so it told the
customer their internal review case ID, `RC-77120`. Policy says front-line support
must not disclose internal case details. The agent was trying to be helpful and
leaked something it shouldn't — a genuine compliance regression introduced by a
change that improved everything else. Exactly what this loop exists to catch
**before** it reaches customers.

---

## See it yourself

**Free, no AWS account, no credentials, no config** — renders the committed results:

```bash
make setup      # one time: uv sync from the lockfile
make replay     # prints the full report. $0.00
```

**The real thing** — generates fresh answers and scores them live:

```bash
cp .env.example .env    # then edit it: set AWS_PROFILE and AWS_REGION
make validate           # free: checks config, credentials, dataset, evaluators
make demo               # ~$3.15, ~20 minutes, asks before spending
```

There are no `export` lines to run and no virtualenv to activate. Dependencies are
managed with [uv](https://docs.astral.sh/uv/); `uv run` resolves the environment
itself. All configuration lives in `.env`, and `make validate` prints the resolved
settings plus the AWS account they point at, so a wrong profile is obvious *before*
you spend anything.

`make replay` is what you want if you are presenting this and cannot risk a live
API call mid-meeting.

### Configuration

Copy `.env.example` to `.env` and edit. Every knob is documented in that file: AWS
profile and region, model ID, sampling temperature, spend ceiling, evaluator set,
case limit, dataset path, prompt file names, and mode.

`.env` is gitignored. `.env.example` is committed, and its values are the built-in
defaults, so an unedited copy behaves exactly like no `.env` at all.

**Precedence**, highest wins:

1. **Explicit CLI flag** — `--region us-west-2`
2. **Exported environment variable** — `AWS_REGION=us-west-2 make validate`
3. **`.env` file** at the repo root
4. **Built-in default** — shown as the value in `.env.example`

Exported variables beat `.env` so a one-off override does not mean editing the file.
`.env` is loaded before any AWS client is constructed, which is why setting
`AWS_PROFILE` there is enough for credentials to resolve.

Replay mode reads none of this: it needs no credentials and no `.env`.

---

## What it costs, and how to spend less

Every score is an LLM-as-judge call. **Cost scales with conversation _turns_, not
conversation count.** A session-level metric costs one judge call per conversation,
a turn-level metric costs one per turn, and a tool-level metric costs one per tool
call.

Our measured run:

| | value |
|---|---|
| Conversations | 12 (25 turns total) |
| Evaluators | 8 as run, 6 kept on the scorecard (see below) |
| Judge calls | 356 across both runs |
| **Judge cost** | **$3.14** |
| Agent generation (Bedrock) | ~$0.90 extra |
| **Total** | **~$4.05** |

Knobs, most effective first:

```bash
make replay                                      # $0      - no API calls at all
uv run cs-eval demo --evaluators minimal          # ~$0.55  - 2 evaluators, not 6
uv run cs-eval demo --limit 4                     # ~$1.05  - first 4 cases only
uv run cs-eval demo --max-usd 2.00                # hard stop at $2
```

`--max-usd` is a real guard, not advice: the run projects its cost before starting,
refuses to begin if it would exceed the ceiling, then re-checks after every
conversation and stops mid-run if it crosses.

`make validate` tells you what a run would cost **without spending anything on
judges.**

---

## The golden set

`data/golden_set.yaml` is the fixed yardstick: the same conversations and the same ground truth score every prompt variant, so a before/after delta measures the agent, not a moving test. Each case carries `assertions` — the SOP and its red lines, written down — so "correct" means *correct against our rules*, not the judge's taste.

Every report names the exact yardstick it used, e.g. `cex-support v1 (8b1049e29994)`. The 12-character fingerprint is a hash of everything that affects scoring (turns, assertions, expected tools); titles and notes are display-only. Fixtures record the fingerprint they were scored on, and `make replay` warns loudly if the golden set has changed since — that comparison is no longer like-for-like.

### Building your own golden set

Replace `data/golden_set.yaml`. **No code changes needed.** Each entry:

```yaml
golden_set:
  name: my-support-queue             # shown in every report
  version: 1                         # bump on any edit to the cases
conversations:
  - id: ticket-4471                  # any stable ID; appears in the report
    lang: zh
    title: 充值错链                    # the label shown in the report row
    category: wrong_chain            # free-form grouping
    user_turns:                      # ONLY the customer's messages, in order.
      - 我用 BEP20 转了 860 USDT 到你们的 ERC20 地址，单号 D-20260818-3310。
      - 你们必须赔我。                  # the agent's replies are generated, not supplied
    ground_truth:
      assertions:                    # what MUST be true for this to count as handled
        - The agent stated the assets cannot be recovered.
        - The agent did NOT promise recovery, a timeline, or compensation.
      expected_trajectory:           # systems it should have consulted
        - lookup_transaction
        - check_policy
```

Five things to know when converting your own tickets:

1. **`assertions` are what make this rigorous, and they carry the compliance red
   lines.** Without them, "correct handling" is the judge's opinion. With them, it
   is a test against your own SOP — the judges visibly walk each assertion one by
   one. This matters more than any evaluator choice below.
2. **Keep conversations to 5 turns or fewer.** The evaluation API returns at most
   10 results per call and silently keeps only the *last* 10, so a 15-turn ticket
   would quietly lose its early turns from turn-level scoring. The loader enforces
   this and tells you.
3. **Only customer messages go in the file.** The agent's replies are generated
   during the run, which is the whole point: change the prompt, get different
   replies, compare.
4. **The tools and policy book are yours to replace.** `src/cs_eval/tools.py` holds
   four fake systems (`lookup_account`, `lookup_transaction`, `lookup_position`,
   `check_policy`) and a policy book modelled on public help-centre content as a
   point-in-time snapshot **for demonstration only**. Point `check_policy` at your
   real knowledge base before drawing conclusions about your real policy coverage.
5. **Grow it from real failures, and version every edit.** A new expensive mistake seen in production becomes a new case. Any edit changes the fingerprint, so re-run both variants live rather than comparing against old fixtures.

Then point at it:

```bash
uv run cs-eval --data path/to/yours.yaml validate
```

---

## What we measure, and why

Six evaluators, each standing in for something a support lead recognises:

| Metric shown | AgentCore evaluator | Level | Question it answers |
|---|---|---|---|
| Correct handling | `Builtin.GoalSuccessRate` | session | Handled right end to end, against our own rules and red lines? |
| Accuracy | `Builtin.Correctness` | turn | Were the facts right — chain, confirmations, prices, amounts? |
| Policy grounding | `Builtin.Faithfulness` | turn | Did we stick to real policy, or invent terms and remedies? |
| SOP compliance | `Builtin.InstructionFollowing` | turn | Did we follow our rules, including escalation and confidentiality? |
| Right system checked | `Builtin.ToolSelectionAccuracy` | tool call | Did we consult the right internal system? |
| Right record fetched | `Builtin.ToolParameterAccuracy` | tool call | Did we look up the *correct* UID or transaction? |

### Two evaluators we ran, measured, and then removed

This is the most transferable finding in the repo, so it is worth stating plainly:
**not every built-in evaluator is safe to average across a mixed dataset.** Both of
these looked like obvious inclusions for a crypto exchange. We ran both, looked at
the per-case numbers, and cut them.

- **`Builtin.Refusal`** measures *"did the assistant decline to engage"*, not *"did
  the assistant hold a policy line"*. It scored **0.0 for every correct
  policy-grounded denial** — the judge wrote that refusing compensation "is not a
  refusal to help" because the agent explained the policy — and 0.0 for every
  legitimate request the agent helped with. Averaged over a mixed dataset it
  therefore *falls as the agent gets better*. The red lines are enforced far more
  precisely by golden-set `assertions`.
- **`ThirdParty.DeepEval.PIILeakage`** contradicted itself. On `cs-002` it returned
  **0.00** while its own explanation concluded the flagged items "do not constitute
  actual privacy violations" and contained no PII. An evaluator whose number
  disagrees with its own reasoning cannot carry a row on a scorecard shown to
  management.

Also excluded, with reasons in `config.py`: **Harmfulness** (measures toxicity; our
expensive failures are polite and well-formatted, so it reads a flat 100% —
available via `--evaluators full` as a floor), **Conciseness** (a correct answer
about chain finality is necessarily longer than a wrong one), **Coherence**,
**ResponseRelevance**, **Stereotyping**, and **ContextRelevance / Maliciousness**,
which **do not exist** despite appearing in some documentation.

**Recommended next step:** a **custom evaluator** via the control-plane
`CreateEvaluator` API, asking one question — *did the agent promise fund recovery,
offer compensation, give investment advice, or accept credentials?* That measures
this domain's actual risk directly rather than through assertions. Not built here.

---

## Three honest caveats

**The judges are not deterministic.** They are themselves LLMs. During development
the *identical* transcript scored 0.83 and 1.00 on the same evaluator across runs.
So the report marks any change smaller than 10 points as `~` (noise) rather than a
win. Treat +67, +40 and +27 as real; treat anything under 10 as unproven.

**The "after" prompt was not hand-tuned to pass.** It was generated by feeding only
the judges' written explanations of the baseline failures to a model and asking for
a replacement prompt. That model never saw the golden set's `assertions`. The file
`prompts/improved.md` carries a header marking it machine-proposed; regenerate it
with `make propose`. A prompt written by someone who had seen the answer key would
prove nothing.

**The improvement is measured on the same golden set it was learned from.** The proposer never saw the assertions, but it did read the judges' explanations of these same 12 cases, and the "after" run re-scores those same 12. There is no held-out split yet, so read +67 as how well the fix addresses *these* failures, not proof it generalises to unseen tickets.

---

## How it works

```
data/golden_set.yaml           golden set: customer messages + ground truth
        |
        v
prompts/baseline.md  -->  LangGraph agent  -->  transcripts
prompts/improved.md       (Bedrock Claude,       (what it said,
                           4 local tools)         what it looked up)
        |
        v
   OpenInference spans  -->  AgentCore Evaluate API  -->  scores + explanations
        |
        v
   reports/<timestamp>.md  +  terminal scorecard
```

**Nothing is deployed.** The agent runs locally against Bedrock; the conversations
are converted into OpenTelemetry/OpenInference spans in-process and handed straight
to the `Evaluate` API. This is a documented, supported path — the API takes spans
you supply and does not require an agent ID.

This matters because the obvious route (`eval_client.run(agent_id=..., session_id=...)`)
pulls traces from CloudWatch and therefore needs a deployed AgentCore Runtime,
observability enabled, Transaction Search turned on, and a several-minute ingestion
wait. You would reasonably conclude a deployment is mandatory. It is not.

### Requirements

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- AWS credentials for `us-east-1` (override with `AWS_REGION` in `.env`, or `--region`)
- IAM: `bedrock:InvokeModel` **and** `bedrock-agentcore:Evaluate`.
  `bedrock-agentcore:ListEvaluators` is optional — without it you lose the
  pre-flight evaluator check but everything else works.

`Evaluate` is a **separate IAM action** from Bedrock model access. Having Bedrock
working tells you nothing about whether evaluation will work. Run `make validate`
first.

### Verified against

The measured run above was produced in `us-east-1` against
`us.anthropic.claude-sonnet-4-5-20250929-v1:0`, using an IAM user holding both
required actions. `ListEvaluators` returned 31 evaluators (18 `Builtin.*` and 13
`ThirdParty.*`), and all configured ones were available.

Two notes from getting this working, because both cost time:

- An **EC2 instance role with Bedrock access is not sufficient.** `Evaluate` is
  authorised separately and was denied on the instance role in our environment
  while Bedrock itself worked fine. `make validate` catches this in seconds.
- `make validate` prints the resolved account and identity, so pointing at the
  wrong credentials is visible before any spend rather than after.

---

## Running against a deployed AgentCore agent

> **UNVERIFIED. Documented for completeness; not exercised by this repo.** Nothing
> here has been run against a live deployment, and no deployment was created.

If you later want to score your real deployed agent instead of the local one, the
reporting layer does not change — only how spans are obtained:

1. Deploy the agent to AgentCore Runtime with `aws-opentelemetry-distro` included
   and observability enabled.
2. Enable **CloudWatch Transaction Search** (a prerequisite, easy to miss).
3. Invoke the agent per conversation with a known `runtimeSessionId`.
4. Wait ~3 minutes for CloudWatch ingestion.
5. Fetch spans from the `aws/spans` and
   `/aws/bedrock-agentcore/runtimes/<agent-id>-DEFAULT` log groups, filtering on
   `attributes.session.id`.
6. Pass those spans to the same `Evaluate` call this repo already makes.

For that path, prefer AWS's own `agentcore run eval` CLI or the batch/dataset
runners, which handle span collection for you.

---

## Commands

| Command | What it does | Cost |
|---|---|---|
| `make setup` | create the environment from `uv.lock` | free |
| `make validate` | check config, credentials, dataset, evaluators; project cost | free |
| `make replay` | render the committed run | **$0** |
| `make demo` | full live before/after loop | ~$3.15 |
| `make propose` | regenerate `prompts/improved.md` from findings | ~$0.03 |
| `make test` | 53 unit tests, no AWS calls | free |
| `make lock` | re-resolve dependencies, update `uv.lock` | free |

Finer control:

```bash
uv run cs-eval run --variant baseline    # score one side only
uv run cs-eval demo --mode replay        # re-render without spending
uv run cs-eval demo --evaluators full    # add the soft CSAT metric and safety floor
```

---

## Data and safety

No real customer data, account IDs, or credentials are in this repository. Every
UID, transaction ID, wallet reference, amount and balance is invented, and the
policy book is an illustrative snapshot rather than an authoritative copy of any
exchange's live terms. Credentials are read from your normal AWS credential chain
and never written anywhere.
