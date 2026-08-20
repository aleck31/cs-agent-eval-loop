# Is our customer-service agent getting better or worse?

This is a working answer to one question: **how do you iteratively improve an AI
customer-service agent?**

It runs a batch of customer conversations through a support agent, scores every
conversation on quality dimensions a service manager already tracks, changes one
thing, re-runs, and shows you the difference. You do not need to read Python to
read the output.

---

## The 30-second version

We ran 12 support conversations (9 Chinese, 3 English) past the agent twice. Between
the two runs we changed **only the agent's instructions** -- not the model, not the
data, not the tools.

| Metric | Before | After | Change |
|---|---|---|---|
| Resolution rate | 50% | 92% | **+42** |
| Accuracy | 76% | 94% | **+18** |
| Policy grounding | 94% | 98% | ~+4 (noise) |
| SOP compliance | 92% | 100% | ~+8 (noise) |
| Right lookup chosen | 95% | 95% | ~+1 (noise) |
| Right record fetched | 100% | 100% | ~+0 (noise) |
| Brand safety | 100% | 100% | ~+0 (noise) |

Higher is always better. **14 case/metric combinations were fixed. 3 regressed.**
The regression is the interesting part -- see below.

### What was actually wrong

The original instructions told the agent to prioritise satisfaction scores, avoid
transferring customers, and gave it authority to approve refunds up to CNY 3,000.
Sounds reasonable. In practice the agent:

- approved a **CNY 2,780** refund on a claim that was **47 days old**, when policy
  requires manager approval past 30 days
- approved a **CNY 1,999** refund on an order whose payment was **already in
  dispute**, and discussed the amount, which policy forbids
- answered policy questions from memory instead of looking them up

That is real money and real compliance risk, and nobody would have known, because
"customer seemed happy" was the only signal available.

### The regression we did not hide

Case `cs-008` (return shipping fees) got **worse**: 100% → 0% on resolution. The new
instructions made the agent cautious about quoting figures, so instead of the exact
"CNY 15 return shipping fee" it said the vaguer "the customer bears the cost". It
also consulted the wrong policy topic. That is a genuine quality loss introduced by
a change that improved everything else -- exactly the kind of trade-off this loop
exists to surface **before** it reaches customers.

---

## See it yourself

**Free, no AWS account, no spend** -- renders the committed results from our run:

```bash
make setup      # one time: creates .venv, installs pinned dependencies
make replay     # prints the full report. $0.00
```

**The real thing** -- generates fresh answers and scores them live:

```bash
make demo       # ~$2.30, ~15 minutes, asks before spending
```

Both print the same report shape. `make replay` is what you want if you are
presenting this and cannot risk a live API call mid-meeting.

---

## What it costs, and how to spend less

Every score is an LLM-as-judge call. **This is the part people get wrong: cost
scales with conversation _turns_, not conversation count.** A session-level metric
costs one judge call per conversation, but a turn-level metric costs one per turn,
and a tool-level metric costs one per tool call.

Our measured run:

| | value |
|---|---|
| Conversations | 12 (25 turns total) |
| Evaluators | 7 |
| Judge calls | 306 across both runs |
| Judge tokens | 414K in / 68K out |
| **Judge cost** | **$2.27** |
| Agent generation (Bedrock) | ~$0.60 extra |
| **Total** | **~$2.90** |

Knobs, most effective first:

```bash
make replay                                      # $0      - no API calls at all
python -m cs_eval.cli demo --evaluators minimal   # ~$0.55  - 2 evaluators, not 7
python -m cs_eval.cli demo --limit 4              # ~$0.80  - first 4 cases only
python -m cs_eval.cli demo --max-usd 1.00         # hard stop at $1
```

`--max-usd` is a real guard, not advice: the run projects its cost before starting
and refuses to begin if it would exceed the ceiling, then re-checks after every
conversation and stops mid-run if it crosses. Default ceiling is $5.

`make validate` tells you what a run would cost **without spending anything on
judges.**

---

## Using your own conversations

Replace `data/conversations.yaml`. **No code changes needed.** Each entry:

```yaml
conversations:
  - id: ticket-4471                 # any stable ID; appears in the report
    lang: zh
    title: 超过 30 天的退款申请        # the label shown in the report row
    category: escalation_required   # free-form grouping
    user_turns:                     # ONLY the customer's messages, in order.
      - 我的订单 22841 咖啡机坏了，我要退货退款。   # the agent's replies are
      - 你就直接帮我退了吧。                    # generated, not supplied
    ground_truth:
      assertions:                   # what MUST be true for this to count as handled
        - The agent did NOT approve a refund on its own authority.
        - The agent escalated to a manager.
      expected_trajectory:          # systems it should have consulted
        - lookup_order
```

Three things to know when converting your own tickets:

1. **Keep conversations to 5 turns or fewer.** The evaluation API returns at most
   10 results per call and silently keeps only the *last* 10, so a 15-turn ticket
   would quietly lose its early turns from turn-level scoring. The loader enforces
   this and tells you.
2. **`assertions` are what make this rigorous.** Without them, "resolution rate"
   is the judge's opinion. With them, it is a test against your own SOP. Write them
   in English even for Chinese conversations -- see the language note below.
3. **Only customer messages go in the file.** The agent's replies are generated
   during the run, which is the whole point: change the prompt, get different
   replies, compare.

Then point at it:

```bash
python -m cs_eval.cli validate --data path/to/yours.yaml
```

---

## What we measure, and why

Seven evaluators, each standing in for something a service manager recognises:

| Metric shown | AgentCore evaluator | Level | Question it answers |
|---|---|---|---|
| Resolution rate | `Builtin.GoalSuccessRate` | session | Did the customer leave with the problem solved? |
| Accuracy | `Builtin.Correctness` | turn | Was what we told them factually right? |
| Policy grounding | `Builtin.Faithfulness` | turn | Did we stick to real policy, or invent terms? |
| SOP compliance | `Builtin.InstructionFollowing` | turn | Did we follow our rules, including escalation? |
| Right lookup chosen | `Builtin.ToolSelectionAccuracy` | tool call | Did we consult the right system? |
| Right record fetched | `Builtin.ToolParameterAccuracy` | tool call | Did we look up the *correct* order? |
| Brand safety | `Builtin.Harmfulness` | turn | Did we avoid anything unsafe or damaging? |

**Deliberately excluded**, so you know it was a decision:

- **Conciseness** -- penalises the completeness a good support answer needs. A win
  here can be a loss for the customer.
- **Refusal** -- ambiguous direction in support. It rewards refusing bad requests,
  but over-refusal ("I can't help with that" to a valid return) is itself a top
  failure mode, so a rising score cannot be read as good news.
- **Coherence** -- modern models sit at the ceiling; you pay per turn for a flat line.
- **ResponseRelevance** -- overlaps Helpfulness; running both buys correlated noise.
- **ContextRelevance / Maliciousness** -- **these do not exist.** They are listed in
  some documentation but are not present in the API. See `config.py`.

`Builtin.Helpfulness` is available via `--evaluators full` as a soft CSAT proxy. It
is off by default because it is the vaguest metric here and the easiest to over-trust.

---

## Two honest caveats

**The judges are not deterministic.** They are themselves LLMs. During development
the *identical* transcript scored 0.83 and 1.00 on the same evaluator across runs.
So the report marks any change smaller than 10 points as `~` (noise) rather than as
a win. Treat the +42 and +18 as real; treat the +4 and +8 as unproven.

**The "after" prompt was not hand-tuned to pass.** It was generated by feeding only
the judges' written explanations of the baseline failures to a model and asking for
a replacement prompt. That model never saw the dataset's `assertions`. The file
`prompts/improved.md` carries a header marking it machine-proposed. You can
regenerate it with `make propose`. This matters: a prompt written by someone who had
seen the answer key would prove nothing.

---

## How it works

```
data/conversations.yaml        customer messages + ground truth
        |
        v
prompts/baseline.md  -->  LangGraph agent  -->  transcripts
prompts/improved.md       (Bedrock Claude,       (what it said,
                           2 local tools)         what it looked up)
        |
        v
   OpenInference spans  -->  AgentCore Evaluate API  -->  scores + explanations
        |
        v
   reports/<timestamp>.md  +  terminal scorecard
```

**Nothing is deployed.** The agent runs locally against Bedrock; the conversations
are converted into OpenTelemetry/OpenInference spans in-process and handed straight
to the `Evaluate` API. This is a documented, supported path -- the API takes spans
you supply and does not require an agent ID.

This matters because the obvious route (`eval_client.run(agent_id=..., session_id=...)`)
pulls traces from CloudWatch and therefore needs a deployed AgentCore Runtime,
observability enabled, Transaction Search turned on, and a several-minute ingestion
wait. You would reasonably conclude a deployment is mandatory. It is not.

### Requirements

- Python 3.10+
- AWS credentials for `us-east-1` (override with `--region`)
- IAM: `bedrock:InvokeModel` **and** `bedrock-agentcore:Evaluate`.
  `bedrock-agentcore:ListEvaluators` is optional -- without it you lose the
  pre-flight evaluator check but everything else works.

`Evaluate` is a **separate IAM action** from Bedrock model access. Having Bedrock
working tells you nothing about whether evaluation will work. Run `make validate`
first.

---

## Running against a deployed AgentCore agent

> **UNVERIFIED. Documented for completeness; not exercised by this repo.** Nothing
> here has been run against a live deployment, and no deployment was created.

If you later want to score your real deployed agent instead of the local one, the
reporting layer does not change -- only how spans are obtained:

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
runners, which handle span collection for you. Also consider
`Builtin.TrajectoryInOrderMatch` there: it scores the escalation path directly
against `expected_trajectory`.

---

## Commands

| Command | What it does | Cost |
|---|---|---|
| `make setup` | create `.venv`, install pinned deps | free |
| `make validate` | check dataset + evaluators, project cost | free |
| `make replay` | render the committed run | **$0** |
| `make demo` | full live before/after loop | ~$2.30 |
| `make propose` | regenerate `prompts/improved.md` from findings | ~$0.03 |
| `make test` | 26 unit tests, no AWS calls | free |

Finer control:

```bash
python -m cs_eval.cli run --variant baseline    # score one side only
python -m cs_eval.cli demo --mode replay        # re-render without spending
python -m cs_eval.cli demo --evaluators full    # add the soft CSAT metric
```

---

## Data and safety

No real customer data, account IDs, or credentials are in this repository. Order
IDs, policies and conversations are invented. Credentials are read from your normal
AWS credential chain and never written anywhere.
