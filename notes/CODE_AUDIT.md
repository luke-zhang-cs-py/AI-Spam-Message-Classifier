# Code audit

## 2026-10-05: third pass

The third pass over the checklist. Suite: **152 pass, 1 skip before, in
12 min 06 s; 157 pass, 1 skip after, in 1 min 50 s** (3 min 20 s
under coverage; with the optional encoder present). Each new regression test failed on the code as
it was, except the threshold equivalence test, which guards a rewrite.

```bash
python -m coverage run --branch -m pytest
python -m coverage report -m
python tools/refresh_figures.py      # keeps docs/index.html and README true
```

### Bugs fixed

| # | Class | Where | What happened | Fix | Test |
|---|---|---|---|---|---|
| 1 | functional / workflow | `spamlib.load_data` | A `path` that held no corpus fell through to the default fallback. `python -m cli.train_spam_classifier --data typo.csv` found no shards at the typo, loaded `dataset.csv` (81 messages), trained, saved over the real model and exited 0. The same happened for an empty directory. | The `dataset.csv` fallback applies only when no path is given. A named path with no CSV raises `FileNotFoundError` naming that path, and the trainer exits 1. | `test_a_data_path_that_holds_no_corpus_is_an_error_not_the_default` |
| 2 | security (CSRF) | `app.py` | `get_json(force=True)` reads a `text/plain` body, which any web site can POST without a preflight. So any page open in the browser could press `/api/retrain` in a loop, pinning a core and overwriting the saved model each time. | A `before_request` hook refuses state-changing requests with a foreign or `null` Origin, or with `Sec-Fetch-Site: cross-site`. This is the guard the face-recognition project in this family already has. | `test_a_cross_site_page_cannot_press_retrain`, `test_the_page_itself_can_still_post` |
| 3 | security (DNS rebinding) | `app.py` | A hostile name that re-resolves to 127.0.0.1 is same-origin with the server, so it could read every reply, including the classified text of pasted messages. | Requests whose Host is not 127.0.0.1, localhost or [::1] get a 403. | `test_a_rebound_hostname_reads_nothing` |

### Performance (perfective)

`choose_threshold` called sklearn's `precision_score`, `recall_score` and `f1_score` for each of about 400 candidate cuts, once per model. That was about 5 s per model and 15 s per comparison. Every training run, every Retrain press and most of the suite's 12 minutes went on input validation inside those calls. The function now counts true positives from two sorted arrays and computes the same ratios sklearn does (`tp/flagged`, `tp/positives`, `2tp/(positives+flagged)`). A comparison now takes under a second.

The result is identical. 450 random cases (ties, NaN scores, no spam at all, unreachable floors) matched the old function exactly, and the chosen MultinomialNB cut on the real corpus is still `0.4575711671752544`, the one `docs/app/js/model-data.js` was built with. `test_the_counted_threshold_search_agrees_with_sklearn` keeps the old loop as the reference.

### Checklist

- **Dispensables.** Nothing new. The long docstrings are this repo's style and they are accurate.
- **Bloaters.** `spamlib.py` (647 lines) is the whole pipeline in one module by design. Its functions are short. Left.
- **Abusers / conditional complexity.** `clean_batch` is already a list of rules. Nothing to fix.
- **Couplers.** `spam_classifier_all_in_one.py` and `cli/train_spam_classifier.py` re-export `spamlib` through wrappers so that the tests can monkeypatch the paths. That is a middle man kept on purpose, and documented. Left.
- **Change preventers.** The loopback guard is now written in two repos of this family. It is small, and sharing it would mean a package. Left.
- **Global data / magic numbers / names.** The constants are named with their reasons (`PRECISION_FLOOR`, `EMBEDDING_C`, `MAX_BODY_BYTES`). Nothing new.
- **Out of bounds.** An empty score vector, NaN scores and a corpus with no spam are all tested in `choose_threshold`. An empty batch, a non-list batch and non-string entries are tested in `clean_batch`.
- **XSS.** The page's `innerHTML` writes pass every server string (token, message) through `esc()`. Labels are fixed strings. Nothing found.
- **Secrets / PII.** None in tracked files. The corpus is synthetic text. `.gitignore` covers `*.joblib`, `models/` and `.coverage`.
- **Left, noted.** `--classify ""` is falsy, so it trains and runs the demo instead of classifying an empty message. That is harmless. `load_data` keeps the first label of a message duplicated across shards with conflicting labels. Nothing in the corpus does this today.

### Coverage (after; `coverage run --branch`)

| File | Lines | Branches |
|---|---|---|
| `app.py` | 100% (147) | 100% (44) |
| `cli/classify.py` | 100% (29) | 100% (6) |
| `cli/train_spam_classifier.py` | 100% (34) | 100% (6) |
| `pipeline/embeddings.py` | 100% (80) | 100% (16) |
| `pipeline/spamlib.py` | 100% (217) | 100% (42) |
| `spam_classifier_all_in_one.py` | 100% (64) | 100% (20) |
| **Total** | **100% (571)** | **100% (134)** |

`tools/` is omitted by `.coveragerc` (developer scripts). The browser build (`docs/app`) is checked by `test_published_figures.py` and the build tool. It is not driven in a browser.

### Maintenance types

- **Corrective:** bug 1.
- **Adaptive:** nothing needed on scikit-learn 1.9, pandas 3.0 and Flask 3.1. `candidate_models` already avoids the deprecated `SVC(probability=True)`.
- **Perfective:** training is about 15x faster, so Retrain answers in about a second.
- **Preventive:** the cross-site and rebinding guards (bugs 2 and 3), and the sklearn-equivalence test.

### Left for later

- `test_the_real_encoder_produces_the_documented_width` takes about 80 s when the encoder is installed. It is now most of the suite's run time.

---

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

## Second audit

**138 tests, 100% of 525 statements before; 153 tests, 100% of 538 after.**
Every fix below has a test that failed against the code before it -- checked
by stashing the source and running the new tests (15 of 15 red).

**Logical -- leakage between folds.** `compare_models` and `metrics`
vectorised the whole corpus once and handed that matrix to
`cross_val_predict`, so the vocabulary and idf of every fold had been
learned partly from the messages it scored. `fold_features` now refits the
extractor inside each training fold. The winner's F1, precision and recall
are unchanged (0.850 / 0.900 / 0.804); its cut moved 0.4559 -> 0.4576;
logistic regression rose 0.761 -> 0.777; and the embedding backend rose
0.853 -> 0.876, which turns "worth one message" into "worth eight". It
stays off by default -- two gigabytes, and not exportable to the browser
build -- but that is now a decision about cost, not about the margin.
Guard: appending a word to one message must not move that message's score.

**Functional -- the page named every model "Thresholded".**
`model_display_name` printed the wrapper's class, and the browser build
published it.

**Runtime -- a JSON body that is not an object was a 500.** `[]`, `"hi"`,
`7` passed `get_json() or {}` and `body.get` raised. `json_body()` fixes
both routes; the browser build now replays those bodies against both sides.

**Runtime -- one damaged artifact broke the app.** joblib reports garbage
as `KeyError`, outside the loader's five-type list, so an interrupted save
meant 500 on every request instead of a retrain. `cli/classify.py` had its
own loader (duplicate code) catching `FileNotFoundError` only; it delegates.

**Security -- request size.** The character caps ran after the whole body
had been read and parsed. `MAX_CONTENT_LENGTH` sits just above the largest
legal batch in its most expensive JSON encoding; Flask answers 413 beyond.

**Dispensables.** `TEST_SIZE` (dead since `evaluate()` moved to
cross-validation, and described on the published page as "the quarter the
web UI scores against") is gone, replaced in the shared-constant checks by
`CV_FOLDS`, which is what the two sides really have to agree on. The dead
`partition` in `cli/classify.classify`. A side-panel note describing a
"held-out split" that did not exist, stale docstring figures in `spamlib`,
and a coverage sentence that read "the uncovered 0 statements" at 100%.
