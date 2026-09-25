# Code audit

First audit of this repository. It was the last of the five projects in this
family without one, and it turned out to be the interesting case: the previous
pass over it had left a fix half-applied, a comment claiming the fix was
complete, and a test that could not fail.

```bash
pytest -q --cov=. --cov-report=term-missing
python -m flake8 . --select=E9,F63,F7,F82,F401,F402,F811,F841,E722,E741,C901 --max-complexity=12
python -m radon cc . -s -n C --exclude "tests/*"
python -m radon mi . -s --exclude "tests/*"
```

## Coverage

**52 tests, 80%.**

| Module | Cover | Uncovered |
|---|---|---|
| `app.py` | 93% | two guard branches, `__main__` |
| `train_spam_classifier.py` | 86% | `main()` |
| `spam_classifier_all_in_one.py` | 71% | `run_interactive`, `main()` |
| `classify.py` | 58% | the interactive `input()` loop |

Maintainability index is A on every module (59.5 – 83.4). flake8 is clean.

**The coverage report was previously measuring the wrong thing, in two ways.**

There was no `.coveragerc`, so the test files were counted as covered source
and padded the total: **79% reported against 68% of the actual modules.**
Every other project in this family omits `tests/`. A coverage figure that
includes its own tests flatters itself by exactly the amount of test code you
have written.

Worse, the figure **depended on whether a gitignored file existed.**
`spam_model.joblib` is build output, so on a machine that had one `ensure_model`
loaded it and `train_and_evaluate` never ran — 43% for the canonical module.
In a fresh clone it had to train, and the same module measured 71%. The number
you got depended on your machine's history, which makes it useless for
comparison over time.

Both are fixed, and the fix was checked rather than assumed: the report is now
**identical — 373 statements, 73 missed, 80% — in the working tree and in a
fresh clone with no model on disk.**

What remains uncovered is honest: the `input()` loops in the two interactive
CLIs, and the `main()` functions. Those mains *are* tested, by
`subprocess.run` against the real scripts, but coverage cannot see across a
process boundary. The alternative — mocking stdin to chase a percentage —
would test less and claim more.

## Findings

### Dispensables — a comment that had become untrue

`spam_classifier_all_in_one.py` carried this above its split constants:

> both numbers used to be written out separately in both files

Past tense. But `train_spam_classifier.py` still held `test_size=0.25` and
`random_state=42` as literals. An earlier pass had named the constants, pointed
`app.py` at them, written the comment as though finished, and left the trainer
alone.

The comment was the most harmful part. A reader checking whether the
duplication was handled would have read that sentence and stopped. Worse than
no comment, because it actively answers the question wrongly.

### Dispensables — a committed build artifact

`.coverage` was tracked: 53 KB of binary SQLite holding the results of one
local test run. It changes on every invocation, so it shows as a modified file
forever and every commit either carries a meaningless diff or has to remember
to leave it out.

`.gitignore` covered the `.joblib` artifacts carefully, with a good comment
about pickled sklearn objects being version-brittle, and then omitted
`.coverage`, `htmlcov/` and `.pytest_cache/` entirely. The transit project in
this family had the identical file committed by accident and its `.gitignore`
says so in as many words -- the fix was never carried across. Untracked, and
the three entries added.

### Magic Number — four literals that had to agree and nothing checked

Beyond the split, `NGRAM_RANGE`, `MIN_DF`, `stop_words` and `max_iter` were
**never shared at all** — named in the canonical module, written out as
`(1, 2)`, `1`, `"english"` and `1000` in the trainer.

Every value happened to match, so there was no live bug. But `app.evaluate()`
scores a model loaded from disk by rebuilding the trainer's split, and that
quarter is only held-out data if it is the *same* quarter. A split changed in
one file and not the other has the app scoring a model against its own
training data and reporting the inflated number as accuracy — which is the
failure the comment described, still fully available.

The trainer now imports all six from the canonical module. `STOP_WORDS` and
`MAX_ITER` were named in the process, since they were unnamed in both places.

This does not weaken the all-in-one file's self-containment, which is its
stated purpose: it still declares everything itself. It is the *other* modules
that stop restating its recipe, which is what `app.py` already did.

### Unit-level bug — an assertion that could not fail

`test_the_split_is_defined_once` ended with:

```python
assert web.clf.TEST_SIZE == allinone.TEST_SIZE
```

`web.clf` **is** the `allinone` module — verified, `web.clf is allinone` is
`True`. So it compared an attribute to itself. Tautological: no change to any
file could have made that line fail.

And the test never looked at `train_spam_classifier.py` at all, despite being
named "the split is defined once". The name asserted a project-wide property
and the body checked one file.

It now states the real claim as identity (`web.clf is allinone` — that `app.py`
uses the canonical module rather than its own recipe), and checks all six
constants against the trainer with `is` rather than `==`, so two separately
declared constants that happen to be equal do not pass.
`test_the_trainer_states_no_recipe_of_its_own` is the structural half.

Both new guards fail against the pre-audit code, checked by extracting it and
running them: `AttributeError` on the first, a restated literal on the second.

### Bloaters, Couplers, Global Data

**`app._state`** is a module-level dict holding the loaded model — Global
Data, and lock-guarded, which is the right shape for a one-process app. But it
had **no `reset()`**, unlike every sibling in this family (`fxrates.reset`,
`schedule.reset`, `realtime.reset_cache`).

That mattered because of what was untested. `/api/retrain` was the one endpoint
no test touched, and it is the only one that mutates process-wide state:
`force_retrain=True` swaps the model out from under every later request. A test
for it would have leaked into every subsequent test in the session. Both added
— the endpoint is tested and the seam exists.

**`clean_batch` is C(12)** and staying that way. It is a flat sequence of six
validation guards, each with a comment naming the bug it prevents — a bare
string iterated character by character, an int raising `AttributeError` inside
a comprehension, an uncapped batch costing hundreds of megabytes per request.
That is a rule table written as code, not tangled logic.

**The duplicated functions across the two pipelines are deliberate.** Six
functions appear in both `spam_classifier_all_in_one.py` and
`train_spam_classifier.py`, and the all-in-one file exists precisely to be one
self-contained script. Deduplicating them would defeat it. The existing
equality tests — `clean_text` agrees, `predict_message` agrees, the artifact
paths agree — are the documented price of that decision, and they are the
right call. Only the *unguarded* half of the duplication was a finding.

### Inconsistent Naming, Uncommunicative Name

No type-suffixed names anywhere. No missing pytest configuration either, now:
there was no `pytest.ini`, so a stray root-level file matching `test_*.py`
gets collected — which happened during this audit, when a copied file in the
project root collided with the one in `tests/` and stopped collection
outright. `testpaths = tests`, as in the wallet and face projects.

## Bug classes

| Class | Found |
|---|---|
| Syntax | none — flake8 clean |
| Runtime | none new |
| Functional | none — the classifier's behaviour is covered and unchanged |
| Logical | **latent, fixed:** four training parameters duplicated with nothing comparing them |
| Workflow | reviewed — preview/commit, batch caps and retrain all behave |
| Unit-level | **fixed:** a tautological assertion, and a test whose name claimed more than its body checked |
| Integration | **fixed:** `/api/retrain` untested; the CLI `main()` untested |
| Out of bounds | reviewed — `MAX_BATCH` and `MAX_MESSAGE_CHARS` both enforced and tested |
| Security | reviewed, clean |

**Security — reviewed and found in good order**, which is worth saying plainly
rather than inventing a finding. Both `innerHTML` sites in `static/js/app.js`
escape the user-derived parts: `esc(t.token)` and `esc(r.message)`. The
unescaped interpolations are `r.label` (server-generated, from a two-word
vocabulary) and numeric confidences. No `eval`, no `shell=True`, no user input
reaching a subprocess. Message and batch sizes are capped.

One note rather than a finding: `esc` is `s.replace(...)`, so a non-string
would raise `TypeError`. Every caller passes server-generated strings today.

## Maintenance classification

**Corrective** — the tautological assertion and the over-claiming test name;
`/api/retrain` and `classify.main()` going untested; the coverage
configuration measuring test files as source.

**Adaptive** — none this pass. Nothing external changed; scikit-learn's API
and the dataset format are as they were.

**Perfective** — sharing the six training constants, naming `STOP_WORDS` and
`MAX_ITER`, adding `reset()`, adding `.coveragerc` and `pytest.ini`. No
behaviour changed: all 47 pre-existing tests passed before and after, and the
model's reported name and metrics are identical after a forced retrain, which
is the assertion that proves the split and seed did not move.

**Preventive** — `test_the_trainer_states_no_recipe_of_its_own` and the
identity checks in `test_the_split_is_defined_once`, both of which fail against
yesterday's code. `reset()` is preventive too: it exists so the next test to
touch the model cache cannot silently reorder the suite.

The lesson worth keeping from this repository is narrower than "check for
duplication". It is that **a fix and its documentation can be committed
separately in effect** — the comment was written for the state the author
intended, not the state the code reached, and it then hid the remainder for
however long it took somebody to check. A test would not have had that
problem, which is why the guard is now a test.
