"""Build the browser-only copy of the classifier into docs/app/.

    python tools/build_static.py
    python tools/build_static.py --prove     (see "Proving it bites", below)

Why a script rather than a second copy of the page: the page in docs/app/ *is*
the page. Its textarea, its verdict badge, its confidence bar, its token chips
and its sample list all come from the same `templates/index.html`,
`static/css/style.css` and `static/js/app.js` the Flask app serves -- copied
here byte for byte and never edited. A hand-maintained fork of app.js is a
fork that looks like a copy, and it rots the first time somebody fixes a bug
in one of them.

What the build adds is the server, and the reason it can is that there is very
little of one. Training is scikit-learn. *Scoring* is not: the chosen model is
a linear classifier over tf-idf features, so a verdict is a sparse dot product
and a comparison against a threshold. So the fitted vocabulary, the idf
vector, the log-probabilities (or coefficients), the class priors and the
chosen cut are exported to JavaScript, and inference -- and only inference --
is reimplemented there:

    tools/static_src/js/spam.js         clean_text, TfidfVectorizer.transform,
                                        predict_proba, the threshold, and
                                        app.token_weights
    tools/static_src/js/static-api.js   app.py's routes, as a fetch shim
    tools/static_src/js/static-ui.js    the one control this build cannot
                                        honour (Retrain)

The numbers are inlined as a `.js` file rather than fetched as a `.json`,
deliberately: a `fetch()` of a relative URL is blocked by the file:// origin
rules, so a JSON bundle would work on GitHub Pages and fail the moment
somebody double-clicked index.html. A `<script src>` has no such restriction,
so the same directory works both ways -- and the fetch shim matches on the
tail of the path, so it also works from a subdirectory such as
/AI-Spam-Message-Classifier/app/.

--------------------------------------------------------------------------
What this refuses to ship
--------------------------------------------------------------------------
A classifier that is *nearly* the Python one is worse than no classifier at
all: it produces a verdict with a confidence next to it and no way for a
reader to know it is not the verdict the repository's own tests are run
against. The failure modes are quiet by construction -- a regular expression
that means something slightly different in JavaScript, a normalisation step
applied in the wrong order, a tie-break in the token ranking -- and every one
of them shows up as a plausible number rather than an error.

So the port is not trusted. Before a byte is written, this script starts a
real browser, loads the JavaScript it is about to publish, and compares it
against the real Python in three passes over the same corpus: every message in
`data/*.csv`, plus a list of deliberately awkward ones -- empty, whitespace
only, punctuation only, control characters, Unicode digits, Greek final
sigma, titlecase digraphs, emoji, full-width Latin, URLs, short codes, phone
numbers, currency amounts, and shouting.

  0. **the cut.** The threshold `spamlib.choose_threshold` picked, compared
     bit for bit against the one the bundle carries -- and the corpus checked
     for its ability to notice a port that *applies* a different one, which
     is a separate hole and needs a message scoring between the chosen cut
     and the 0.5 a careless port would fall back to. The build refuses if the
     corpus has no such message rather than quietly losing the coverage.

  1. **normalisation.** `spamlib.clean_text` against `SpamModel.cleanText`,
     compared as exact strings. This is where Python's regular expressions and
     JavaScript's disagree most: `\\w`, `\\d`, `\\s` and therefore `\\b` all
     mean different sets of characters in the two languages, and a message
     pasted out of a spreadsheet really does contain U+001C.

  2. **the verdict, the probability and the explanation.** `app.classify`
     against `SpamModel.classify`, field by field: the label (also checked
     against `spamlib.predict_message`, which is the function the CLI calls),
     the confidence, the recognised-token count, the cleaned string, and the
     full ranked list of per-token weights -- `token`, `weight` and `tfidf`,
     in order. The weights are compared because the page *shows* them, and a
     number a reader can see is as much the behaviour as the verdict is. The
     raw P(spam) is compared as well, to a tolerance that is four orders of
     magnitude tighter than the last digit anything displays.

  3. **the routes.** The same HTTP requests, including the ones that must
     fail, replayed against Flask's test client and against the fetch shim,
     with status codes and bodies compared. That is the pass that keeps
     app.js -- byte-identical in both builds and therefore unable to adapt --
     from being able to tell which one it is running in.

Any disagreement stops the build. Nothing is written.

--------------------------------------------------------------------------
Proving it bites
--------------------------------------------------------------------------
A comparison that cannot fail is worse than no comparison, because it reads
like one that passed. `--prove` is the standing answer to that: it breaks the
bundle on purpose, six ways, and requires the checks to catch every one.

    threshold    move the cut by 1e-12          -> pass 0 compares it exactly
    applied-cut  report the cut, apply 0.5      -> pass 2, as flipped verdicts
    idf          nudge one idf weight by 1e-6   -> tfidf, weights, probability
    logprob      nudge one fitted log-prob      -> probability and confidence
    vocabulary   misspell one term              -> the wrong word is explained
    shoutRatio   change when capitals count     -> cleaned text changes

Each is injected into the *published* model-data.js, so it is the real bundle
being broken rather than a mock of it. If any sabotage survives the checks,
`--prove` fails and says which -- that is a hole in the comparison, and it is
reported as a build failure rather than as a passing run.

It has already found one. The first version of this script moved the decision
threshold by 0.02 and nothing noticed, because no message in 556 lands within
two points of the cut: the comparison was covering the arithmetic and not the
number it is compared against. Pass 0 exists because `--prove` failed.

The browser is the system Chrome if it is there and Playwright's Chromium
otherwise. That is a build-time dependency, not a runtime one: `pip install
-r requirements.txt` and `pytest` do not need it, and neither does the
published page. Running the JavaScript rather than transcribing it a third
time is the point -- a transcription of a port is just a third thing to get
wrong.
"""

import contextlib
import gzip
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "docs", "app")
SRC = os.path.join(ROOT, "tools", "static_src")

REPO = "https://github.com/luke-zhang-cs-py/AI-Spam-Message-Classifier"

# Where Chrome lives on the machine this is developed on. Optional: Playwright
# ships its own Chromium and that is used when this is not here, so the build
# works on a runner too.
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# Copied unchanged from the Flask app, so both builds run the same code.
SHARED_JS = ["app.js"]
STATIC_JS = ["spam.js", "static-api.js", "static-ui.js"]

# The order the page loads them in. model-data.js first because spam.js reads
# it at definition time; static-api.js before app.js, which calls loadModel()
# as it finishes; static-ui.js last, because it corrects a control app.js
# binds.
SCRIPT_ORDER = ["js/model-data.js", "js/spam.js", "js/static-api.js",
                "js/app.js", "js/static-ui.js"]

BANNER = "/* Generated by tools/build_static.py -- edit %s instead. */\n"

# How close the two sides have to agree on a raw probability. Everything the
# page displays is rounded to four decimals and compared exactly; this is for
# the unrounded number underneath, and it is loose only in the sense that
# Math.exp and numpy's exp are allowed to differ in the last bit or two.
PROBABILITY_TOLERANCE = 1e-12


def read(path):
    with io.open(path, encoding="utf-8") as handle:
        return handle.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def stop(message):
    raise SystemExit("build_static: " + message)


def replace_once(text, old, new, what):
    """A substitution that fails loudly if the source moved.

    Every edit below is a string match against templates/index.html. A miss is
    silent by default -- the page builds, looks nearly right, and is missing
    the one paragraph that says it is not the Flask app. So each one is
    counted.
    """
    count = text.count(old)
    if count != 1:
        stop("could not %s -- expected exactly one match in "
             "templates/index.html, found %d. The template has moved; fix the "
             "anchor in tools/build_static.py.\n  looking for: %r"
             % (what, count, old[:120]))
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# The awkward messages
# ---------------------------------------------------------------------------
# Chosen, not random. The corpus in data/ is 400-odd ordinary text messages and
# it exercises the happy path thoroughly; none of it contains a lone control
# character, an Arabic-Indic digit or a Greek final sigma, and those are
# exactly where a Python regular expression and a JavaScript one stop meaning
# the same thing. Each line below is here because some specific step of
# clean_text or of the analyser could plausibly diverge on it.
ADVERSARIAL = [
    # Nothing, and nearly nothing.
    "", " ", "\t", "\n", "\r\n", "   \t\n  ", "\v\f",
    # Python counts these as whitespace and JavaScript does not...
    "\x1c\x1d\x1e\x1f", "\x85", "a\x1cb", "\x85free\x85",
    # ...and JavaScript counts this one and Python does not.
    "\ufeff", "\ufefffree", "free\ufeff",
    "\u00a0", "\u00a0free\u00a0", "\u2003free\u2003money\u2003",
    "\u2028\u2029", "\u200b\u200bzero width",
    # Punctuation only. Every character of string.punctuation is stripped, so
    # each of these cleans to the empty string and scores on the priors alone.
    "!", "!!!", "...", "?!?!", "@#$%^&*()", "____", "-", "--", "~", "`",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    # One and two characters: the token pattern needs two, so "a" vectorises
    # to nothing at all and "ab" to one term.
    "a", "ab", "a b", "I", "ok", "OK", "No.", "hi", "Hi!", "HI!!!",
    # Shouting, and the ratio either side of SHOUT_RATIO.
    "FREE", "free", "Free", "FREE FREE FREE", "free free free",
    "ALL CAPS AND NOTHING ELSE AT ALL IN THIS ONE",
    "mixed CASE with SOME caps", "aB", "Ab", "aBc", "ABc", "AbC",
    # Digits, short codes, phone numbers, money -- the four substitutions, and
    # the boundaries between them.
    "123", "1234", "12345", "123456", "1234567", "12345678",
    "12,345", "1.234", "1,2,3", "1.", ".5", "1..2", "0.", "000.10",
    "0712345678", "07123456789", "071234567890", "0712345678901234",
    "Call 07123456789 now", "Text WIN to 80086", "text win to 80086",
    "\u00a3100", "$1,000.50", "\u20ac 9.99", "\u00a3 5", "$5", "\u20ac5,00",
    "$", "\u00a3", "\u20ac", "$x", "\u00a3.", "5%", "100%", "50/50",
    # Links, which is the first and greediest substitution.
    "www.example.com", "http://a.b/c", "https://x.io/?q=1&r=2",
    "visit bit.ly/abc now", "see foo.org/bar", "mail me at a@b.com",
    "HTTP://SHOUTY.COM/PATH", "www.", "a.com", ".com", "x.co/",
    "claim.now.info/free?id=9", "no.dots.here",
    # Unicode letters: accents, the two sigmas, a titlecase digraph, a dotted
    # capital I, Roman numerals and circled capitals (which are uppercase to
    # Python but not letters), full-width Latin, and CJK.
    "caf\u00e9", "CAF\u00c9", "Caf\u00e9 Cr\u00e8me", "na\u00efve",
    "\u0391\u03a3", "\u039f\u03a3", "\u03a3\u0399\u0393\u039c\u0391",
    "\u03c3\u03af\u03b3\u03bc\u03b1",
    "\u0130stanbul", "\u01c5ungla", "\u01c4UNGLA", "\u01c4UNGLA WIN FREE",
    "\u2167 \u2168 \u2169", "\u24b6\u24b7\u24b8",
    "\uff26\uff32\uff25\uff25 \uff2d\uff2f\uff2e\uff25\uff39",
    "\uff11\uff12\uff13\uff14",
    "\u0663\u0664\u0665\u0666", "\u0661\u0662\u0663\u0664\u0665\u0666",
    "\u0660\u0667\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
    "g\u00fcnstig gewinnen JETZT", "\u514d\u8cbb \u4e2d\u734e",
    "a\u0301b", "\u0301combining",
    # Emoji, which are surrogate pairs in JavaScript and one character in
    # Python -- the shouting ratio counts characters.
    "\U0001f389 free money \U0001f389", "\U0001f44d",
    "\U0001f389\U0001f389\U0001f389", "\U0001f389FREE\U0001f389",
    # Underscores, which string.punctuation removes.
    "a_b c_d", "__init__ __main__", "_", "__",
    # Whitespace shapes and length.
    "one\ntwo\nthree", "tab\tseparated", "trailing   ", "   leading",
    "x" * 500, "free " * 100, "FREE " * 100, ("a b " * 300),
    # The five samples the page offers, which are the first thing anyone
    # clicks and therefore the first thing that would be visibly wrong.
    "Congratulations! You've won a $1000 gift card. Click here to claim now",
    "Hey are we still on for lunch tomorrow at noon",
    "URGENT: your account has been suspended, verify your details immediately",
    "Can you review my essay before I submit it tomorrow",
    "Free entry into our weekly draw, text WIN to 80086 now",
]


def corpus():
    """Every message the two sides are compared on."""
    from pipeline import spamlib

    frame = spamlib.load_data()
    return list(frame["text"].astype(str)) + ADVERSARIAL


# The requests replayed against both builds, in order. The refusals are the
# point of most of them: app.js shows `data.error` verbatim in the red box, so
# a shim that worded one differently, or answered 200 where Flask answers 400,
# would be a page that disagrees with the repository about what it just did.
#
# /api/retrain is the one route deliberately absent, and it is the one route
# where the two builds are *meant* to differ: Flask retrains and answers 200,
# the shim answers 400 saying it cannot. Replaying it here would assert they
# match, which would be asserting a lie -- so it is left out, and said out
# loud, rather than quietly passing because nobody looked. What keeps that
# difference from reaching a reader as a dead button is static-ui.js, which
# disables the control and puts the same sentence next to it.
def requests_for(messages):
    long_message = "free " * 5000            # over MAX_MESSAGE_CHARS
    return [
        ("GET", "/api/model", None),
        ("POST", "/api/classify", {"message": messages[0]}),
        ("POST", "/api/classify", {"message": "  " + messages[1] + "  "}),
        ("POST", "/api/classify", {"message": "Free entry, text WIN to 80086"}),
        ("POST", "/api/classify", {"message": "see you at noon"}),
        ("POST", "/api/classify", {"message": "\u00a3500 CASH PRIZE CLAIM NOW"}),
        ("POST", "/api/classify", {"message": "zzzzqqq wwwwxxx"}),
        ("POST", "/api/classify", {"message": ""}),
        ("POST", "/api/classify", {"message": "   "}),
        ("POST", "/api/classify", {"message": "\x1c\x1d"}),
        ("POST", "/api/classify", {"message": "\ufeff"}),
        ("POST", "/api/classify", {"message": None}),
        ("POST", "/api/classify", {"message": 5}),
        ("POST", "/api/classify", {}),
        ("POST", "/api/classify", {"message": long_message}),
        ("POST", "/api/batch", {"messages": messages[:25]}),
        ("POST", "/api/batch", {"messages": ["free money now", "lunch at 1"]}),
        ("POST", "/api/batch", {"messages": ["  spaced  ", "", "   "]}),
        ("POST", "/api/batch", {"messages": []}),
        ("POST", "/api/batch", {"messages": ["", "  "]}),
        ("POST", "/api/batch", {"messages": "free money"}),
        ("POST", "/api/batch", {"messages": ["ok", 5]}),
        ("POST", "/api/batch", {"messages": ["ok", None]}),
        ("POST", "/api/batch", {"messages": None}),
        ("POST", "/api/batch", {}),
        ("POST", "/api/batch", {"messages": ["x"] * 201}),
        ("POST", "/api/batch", {"messages": ["ok", long_message]}),
    ]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def load_model():
    """The fitted (model, vectorizer), trained first if they are not on disk.

    The .joblib artifacts are gitignored -- a pickled estimator is brittle
    across scikit-learn releases -- so a fresh clone has none, and a build
    script that failed there would mean the published page could only be
    rebuilt on a machine that had already run the app. Training is the same
    deterministic, seeded call app.py makes, and it takes about a second.
    """
    from pipeline import spamlib

    model, vectorizer = spamlib.load_artifacts()
    if model is None or vectorizer is None:
        print("  no saved artifacts; training from data/ ...")
        model, vectorizer = spamlib.train_and_evaluate(spamlib.load_data(),
                                                       verbose=False)
        spamlib.save_model(model, vectorizer)
    return model, vectorizer


def collect_model(model, vectorizer):
    """Everything the ported scorer reads, and the checks that it is enough.

    The estimator is inspected rather than assumed. `spamlib.candidate_models`
    offers three and cross-validation picks whichever wins on the corpus of
    the day, so a build that hard-coded naive Bayes would silently ship the
    wrong arithmetic the first time logistic regression won -- and all three
    give plausible-looking probabilities, so nobody would notice from the
    page. All three are ported; anything else stops the build.
    """
    import numpy as np
    import app as flask_app
    from pipeline import spamlib

    if spamlib.wants_raw_text(vectorizer):
        stop("this model was trained with the optional embedding backend, "
             "which is a 22 MB sentence-transformer and 384 dense features "
             "per message. That is not something to inline in a page.\n"
             "  Retrain without --embeddings, or publish the tf-idf model:\n"
             "  python -m cli.train_spam_classifier")

    from sklearn.feature_extraction.text import TfidfVectorizer
    if not isinstance(vectorizer, TfidfVectorizer):
        stop("expected a TfidfVectorizer, found %s. spam.js ports "
             "TfidfVectorizer.transform and nothing else."
             % type(vectorizer).__name__)

    for name, wanted, got in [
            ("analyzer", "word", vectorizer.analyzer),
            ("ngram_range", spamlib.NGRAM_RANGE, tuple(vectorizer.ngram_range)),
            ("lowercase", True, vectorizer.lowercase),
            ("sublinear_tf", True, vectorizer.sublinear_tf),
            ("norm", "l2", vectorizer.norm),
            ("use_idf", True, vectorizer.use_idf),
            ("binary", False, vectorizer.binary),
            ("strip_accents", None, vectorizer.strip_accents),
            ("stop_words", None, vectorizer.stop_words),
            ("preprocessor", None, vectorizer.preprocessor),
            ("tokenizer", None, vectorizer.tokenizer),
            ("token_pattern", r"(?u)\b\w\w+\b", vectorizer.token_pattern)]:
        if got != wanted:
            stop("the vectoriser's %s is %r and spam.js ports %r. Either the "
                 "configuration in spamlib.build_vectorizer changed or this "
                 "artifact predates it; port the change in "
                 "tools/static_src/js/spam.js before rebuilding."
                 % (name, got, wanted))

    estimator = getattr(model, "estimator", model)
    threshold = float(getattr(model, "threshold", 0.5))

    payload = {
        # /api/model reports the wrapper's class name, which is what the page
        # prints next to "Classifier".
        "name": flask_app.model_display_name(model),
        "estimator": type(estimator).__name__,
        "chosen": getattr(model, "name", type(estimator).__name__),
        "threshold": threshold,
        "shoutRatio": spamlib.SHOUT_RATIO,
        "maxMessageChars": flask_app.MAX_MESSAGE_CHARS,
        "maxBatch": flask_app.MAX_BATCH,
        "terms": [str(t) for t in vectorizer.get_feature_names_out()],
        "idf": [float(v) for v in vectorizer.idf_],
    }

    # app.token_weights reads coef_ first and feature_log_prob_ second, and
    # the branch it takes decides what a token's weight means. Recorded here
    # rather than re-derived in the browser, so the two cannot disagree about
    # which kind of model this is.
    if hasattr(model, "coef_"):
        payload["kind"] = "linear"
        payload["coef"] = [float(v) for v in np.asarray(model.coef_).ravel()]
        payload["intercept"] = float(np.asarray(model.intercept_).ravel()[0])
        payload["logProb"] = None
        payload["logPrior"] = None
        payload["addPrior"] = False
    elif hasattr(model, "feature_log_prob_"):
        log_prob = np.asarray(model.feature_log_prob_)
        payload["kind"] = "naivebayes"
        payload["coef"] = None
        payload["logProb"] = [[float(v) for v in row] for row in log_prob]
        payload["logPrior"] = [float(v)
                               for v in np.asarray(model.class_log_prior_)]
        # ComplementNB._joint_log_likelihood does not add the prior when there
        # is more than one class; MultinomialNB always does. The one line of
        # difference between the two, and the only reason this flag exists.
        payload["addPrior"] = type(estimator).__name__ != "ComplementNB"
    else:
        stop("the fitted estimator (%s) exposes neither coef_ nor "
             "feature_log_prob_, so there is no linear weight to export and "
             "no way to explain a verdict. spam.js cannot score it."
             % type(estimator).__name__)

    if list(model.classes_) != [0, 1]:
        stop("classes_ is %r; spam.js assumes the two-class [ham, spam] "
             "ordering that spamlib.LABELS defines." % (list(model.classes_),))

    if payload["logProb"] is not None and len(payload["logProb"]) != 2:
        stop("feature_log_prob_ has %d rows, not 2."
             % len(payload["logProb"]))

    width = len(payload["terms"])
    for name, vector in [("idf", payload["idf"]),
                         ("coef", payload["coef"])]:
        if vector is not None and len(vector) != width:
            stop("%s has %d entries against %d vocabulary terms."
                 % (name, len(vector), width))
    if payload["logProb"] is not None:
        for row in payload["logProb"]:
            if len(row) != width:
                stop("feature_log_prob_ row has %d entries against %d "
                     "vocabulary terms." % (len(row), width))
    return payload


def model_metrics(model, vectorizer):
    """The figures /api/model serves, measured the way app.py measures them."""
    import app as flask_app
    from pipeline import spamlib

    frame = spamlib.load_data()
    return flask_app.evaluate(frame, model, vectorizer), int(len(frame))


def data_file(payload):
    """The exported model as one `const`, with its layout written down.

    ASCII-escaped, which costs nothing today -- the corpus has produced no
    non-ASCII vocabulary term yet -- and is not a style preference. A
    `<script src>` carries no charset of its own: over file:// the browser
    decodes it with whatever it guessed for the document, and a vocabulary
    term decoded as Latin-1 is a term that never matches, which is a changed
    verdict with nothing on the page to show for it.
    """
    body = json.dumps(payload, sort_keys=False, separators=(",", ":"),
                      ensure_ascii=True)
    return (
        "/* Generated by tools/build_static.py from the fitted artifacts --\n"
        " * do not edit. Rebuild with: python tools/build_static.py\n"
        " *\n"
        " * The model, inlined as a script rather than fetched as JSON so\n"
        " * that opening index.html from the filesystem works as well as\n"
        " * serving it: a relative fetch() is blocked by the file:// origin\n"
        " * rules and a <script src> is not.\n"
        " *\n"
        " * Nothing here is rounded. Every float is Python's shortest\n"
        " * round-tripping repr, so JSON.parse recovers the same IEEE double\n"
        " * scikit-learn fitted -- which is what lets the build compare the\n"
        " * two sides exactly rather than approximately.\n"
        " *\n"
        " *   terms       column -> the tf-idf term, in vocabulary order\n"
        " *   idf         column -> TfidfVectorizer.idf_\n"
        " *   logProb     [ham, spam] rows of feature_log_prob_  (naive Bayes)\n"
        " *   logPrior    class_log_prior_\n"
        " *   coef        LogisticRegression.coef_               (linear)\n"
        " *   threshold   the cut spamlib.choose_threshold picked, which\n"
        " *               travels with the model rather than being 0.5\n"
        " *   shoutRatio  spamlib.SHOUT_RATIO\n"
        " *   metrics     what /api/model reports, measured the way app.py\n"
        " *               measures it: out-of-fold over %d messages\n"
        " */\n"
        "const SPAM_MODEL_DATA = %s;\n"
        % (payload["metrics"]["testSize"], body))


# ---------------------------------------------------------------------------
# The browser
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def chromium():
    """One browser for the whole build, checks and proofs alike."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        stop("this build runs the JavaScript it is about to publish against "
             "the Python it was ported from, and that needs a browser.\n"
             "  pip install playwright && python -m playwright install chromium\n"
             "There is no --skip-checks flag on purpose: an unchecked port of "
             "the scorer is the one thing this script exists to prevent.")

    with sync_playwright() as pw:
        options = {}
        if os.path.exists(CHROME):
            options["executable_path"] = CHROME
        instance = pw.chromium.launch(**options)
        try:
            yield instance
        finally:
            instance.close()


@contextlib.contextmanager
def loaded(instance, sources):
    """A page holding the bundle's JavaScript, and no tolerance for errors.

    about:blank, so there is no origin and no server -- which is the same
    situation a file:// double-click puts the page in, and the first thing
    that would notice if the shim ever needed one.
    """
    html = ('<!doctype html><html><head><meta charset="utf-8">'
            "<title>build check</title></head><body>"
            + "".join("<script>\n%s\n</script>" % text for text in sources)
            + "</body></html>")
    page = instance.new_page()
    problems = []
    page.on("pageerror", lambda bad: problems.append(str(bad)))
    page.on("console",
            lambda msg: problems.append(msg.text) if msg.type == "error"
            else None)
    try:
        page.set_content(html)
        if problems:
            stop("the bundle's JavaScript does not load:\n  "
                 + "\n  ".join(problems))
        yield page, problems
    finally:
        page.close()


# ---------------------------------------------------------------------------
# The JavaScript side of each pass
# ---------------------------------------------------------------------------
CLEAN_JS = "(messages) => messages.map((m) => SpamModel.cleanText(m))"

CLASSIFY_JS = """(messages) => messages.map((m) => {
  const out = SpamModel.classify(m);
  out.probability = SpamModel.spamProbability(m);
  return out;
})"""

THRESHOLD_JS = "() => SpamModel.threshold"

ROUTES_JS = """async (requests) => {
  const out = [];
  for (const request of requests) {
    const init = { method: request.method };
    if (request.body !== null && request.body !== undefined) {
      init.headers = { 'Content-Type': 'application/json' };
      init.body = JSON.stringify(request.body);
    }
    const reply = await window.fetch(request.url, init);
    out.push({ status: reply.status, json: await reply.json() });
  }
  return out;
}"""


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------
def _same_leaf(want, got):
    """Whether two values at the bottom of a payload agree.

    Numbers compare across int and float, because JSON has one number type and
    the two sides hand back whichever fits. A bool never compares equal to a
    number, though: `True == 1` in Python, and `{"ok": true}` coming back as
    `{"ok": 1}` is precisely the kind of thing this is for.
    """
    if isinstance(want, bool) != isinstance(got, bool):
        return False
    return want == got


def differences(want, got, path="", found=None):
    """Every place two payloads disagree, with the path to each."""
    if found is None:
        found = []
    if isinstance(want, dict) and isinstance(got, dict):
        for key in sorted(set(want) | set(got)):
            where = "%s.%s" % (path, key)
            if key not in want:
                found.append("%s: only the browser has it (%r)"
                             % (where, got[key]))
            elif key not in got:
                found.append("%s: only python has it (%r)" % (where, want[key]))
            else:
                differences(want[key], got[key], where, found)
    elif isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            found.append("%s: python has %d entries, browser has %d\n"
                         "    python:  %r\n    browser: %r"
                         % (path or ".", len(want), len(got), want, got))
        else:
            for index, (mine, yours) in enumerate(zip(want, got)):
                differences(mine, yours, "%s[%d]" % (path, index), found)
    elif not _same_leaf(want, got):
        found.append("%s: python %r, browser %r" % (path or ".", want, got))
    return found


def report(what, wrong, extra=""):
    if not wrong:
        return
    shown = wrong[:25]
    stop("%s: the browser build disagrees with the Python in %d place%s.\n"
         "Nothing has been written to docs/app.\n%s  %s%s"
         % (what, len(wrong), "" if len(wrong) == 1 else "s", extra,
            "\n  ".join(shown),
            "\n  ... and %d more" % (len(wrong) - len(shown))
            if len(wrong) > len(shown) else ""))


def short(message):
    """A message, quoted, short enough to read in a build log."""
    text = repr(message)
    return text if len(text) <= 70 else text[:67] + "...'"


# ---------------------------------------------------------------------------
# The Python side, measured once
# ---------------------------------------------------------------------------
def python_reference(messages, model, vectorizer):
    """Every answer the real implementation gives, for all three passes.

    Measured once and reused, because --prove runs the comparison six more
    times with the bundle broken in six different ways, and the Python side
    is the same on every one of them. Re-running Flask's test client seven
    times would be most of the wall clock and none of the information.
    """
    import app as flask_app
    from pipeline import spamlib

    payloads = [flask_app.classify(m, model, vectorizer) for m in messages]

    flask_app.reset()
    client = flask_app.app.test_client()
    replies = []
    wanted = [{"method": method, "url": url, "body": body}
              for method, url, body in requests_for(messages)]
    for request in wanted:
        extra = {} if request["body"] is None else {"json": request["body"]}
        reply = client.open(request["url"], method=request["method"], **extra)
        replies.append({"status": reply.status_code, "json": reply.get_json()})

    return {
        "messages": messages,
        "cleaned": [spamlib.clean_text(m) for m in messages],
        "payloads": payloads,
        # The CLI's answer as well as the web app's. They go through the same
        # Thresholded.predict, and this is what says so.
        "verdicts": [spamlib.predict_message(m, model, vectorizer)
                     for m in messages],
        "probabilities": [spamlib.spam_probability(m, model, vectorizer)
                          for m in messages],
        "threshold": float(getattr(model, "threshold", 0.5)),
        "requests": wanted,
        "replies": replies,
    }


# ---------------------------------------------------------------------------
# Pass 0 -- the cut itself
# ---------------------------------------------------------------------------
def check_threshold(page, reference):
    """The exported cut, and whether the corpus can tell if it moved.

    Two separate worries, and the corpus only covers one of them. A port that
    *applied* the wrong cut -- compared against a hard-coded 0.5 instead of
    the number spamlib.choose_threshold picked -- shows up as a flipped
    verdict, but only for a message that scores between the two, so it is the
    corpus that catches it and the corpus has to be checked for the ability.
    A port that carried a slightly *different* cut would need a message
    within that slight difference of it, and no corpus can promise that. So
    the number is compared directly, bit for bit, as well.

    `--prove` breaks it both ways, because these are two different holes.
    """
    published = page.evaluate(THRESHOLD_JS)
    if published != reference["threshold"]:
        stop("the published decision threshold is %.17g and the fitted model's "
             "is %.17g. Nothing has been written to docs/app."
             % (published, reference["threshold"]))

    cut = reference["threshold"]
    below = [p for p in reference["probabilities"] if p < cut]
    above = [p for p in reference["probabilities"] if p >= cut]
    if not below or not above:
        stop("every message in the comparison corpus scores on the same side "
             "of the %.4f cut, so the verdict comparison cannot notice a port "
             "that applies a different one. Add a message that lands on the "
             "other side to ADVERSARIAL in tools/build_static.py." % cut)

    # The band between the chosen cut and the default it replaced is where a
    # port that quietly fell back to 0.5 would differ, so the corpus is
    # required to have something in it.
    if cut < 0.5 and not [p for p in above if p < 0.5]:
        stop("no message in the comparison corpus scores between the chosen "
             "cut (%.4f) and the 0.5 a port would fall back to, so a port "
             "that ignored the exported threshold would pass every check "
             "here. Add one to ADVERSARIAL in tools/build_static.py." % cut)

    return cut, min(cut - p for p in below), min(p - cut for p in above)


# ---------------------------------------------------------------------------
# Pass 1 -- clean_text
# ---------------------------------------------------------------------------
def check_cleaning(page, reference):
    theirs = page.evaluate(CLEAN_JS, reference["messages"])
    wrong = []
    for message, mine, yours in zip(reference["messages"],
                                    reference["cleaned"], theirs):
        if mine != yours:
            wrong.append("%s\n      python  %r\n      browser %r"
                         % (short(message), mine, yours))
    report("spamlib.clean_text vs SpamModel.cleanText", wrong,
           "Python and JavaScript disagree about what \\w, \\d, \\s or \\b "
           "mean here.\n")
    return len(reference["messages"])


# ---------------------------------------------------------------------------
# Pass 2 -- the verdict, the probability and the explanation
# ---------------------------------------------------------------------------
def check_scoring(page, reference):
    theirs = page.evaluate(CLASSIFY_JS, reference["messages"])

    wrong = []
    for index, yours in enumerate(theirs):
        message = reference["messages"][index]
        mine = reference["payloads"][index]

        if reference["verdicts"][index] != mine["label"]:
            wrong.append("%s: predict_message says %r and app.classify says "
                         "%r -- the Python disagrees with itself"
                         % (short(message), reference["verdicts"][index],
                            mine["label"]))

        probability = reference["probabilities"][index]
        gap = abs(probability - yours["probability"])
        if gap > PROBABILITY_TOLERANCE:
            wrong.append("%s: P(spam) python %.17g, browser %.17g (%.3g apart)"
                         % (short(message), probability, yours["probability"],
                            gap))

        for field in ("label", "cleaned", "confidence", "knownTokens"):
            if not _same_leaf(mine[field], yours.get(field)):
                wrong.append("%s: %s python %r, browser %r"
                             % (short(message), field, mine[field],
                                yours.get(field)))

        # The token chips, in the order the page lays them out. Compared
        # whole rather than as a set: the ranking is what the reader treats
        # as "which word mattered most", and a port that got the tie-break
        # backwards would put a different word first.
        wrong += ["%s: %s" % (short(message), line)
                  for line in differences(mine["tokens"],
                                          yours.get("tokens"), "tokens")]

    report("app.classify vs SpamModel.classify", wrong)
    return len(reference["messages"])


# ---------------------------------------------------------------------------
# Pass 3 -- the routes
# ---------------------------------------------------------------------------
def check_routes(page, reference):
    """The same requests, to Flask and to the shim, with the same model."""
    theirs = page.evaluate(ROUTES_JS, reference["requests"])

    wrong = []
    for request, mine, yours in zip(reference["requests"],
                                    reference["replies"], theirs):
        label = "%s %s %s" % (request["method"], request["url"],
                              short(json.dumps(request["body"]))
                              if request["body"] is not None else "")
        wrong += differences(mine, yours, label.strip())
    report("app.py vs static-api.js", wrong,
           "app.js is byte-identical in both builds, so a difference here is "
           "one it cannot adapt to.\n")
    return len(reference["requests"])


# ---------------------------------------------------------------------------
# Running them
# ---------------------------------------------------------------------------
def harness(written):
    """The three files the checks are about, in load order, exactly as they
    will be published -- not the sources they were read from."""
    return [written["js/model-data.js"], written["js/spam.js"],
            written["js/static-api.js"]]


def run_checks(instance, written, reference, loud=True):
    with loaded(instance, harness(written)) as (page, problems):
        cut, under, over = check_threshold(page, reference)
        if loud:
            print("  threshold  == threshold            %.17g, and the corpus "
                  "straddles it" % cut)
            print("                                     (nearest below %.4f "
                  "under, nearest above %.4f over)" % (under, over))
        cleaned = check_cleaning(page, reference)
        if loud:
            print("  clean_text == cleanText            %d messages, compared "
                  "as exact strings" % cleaned)
        scored = check_scoring(page, reference)
        if loud:
            print("  classify   == classify             %d messages: verdict, "
                  "P(spam), confidence," % scored)
            print("                                     recognised-token "
                  "count, cleaned text and every token weight")
        routes = check_routes(page, reference)
        if loud:
            print("  app.py     == static-api.js        %d requests, replayed "
                  "against both" % routes)
        if problems:
            stop("the bundle's JavaScript logged errors while being "
                 "checked:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Proving the checks bite
# ---------------------------------------------------------------------------
# Each of these is one line of JavaScript appended to the *published*
# model-data.js, where it runs before spam.js reads it. So it is the real
# bundle being broken, not a mock of it, and the pass that is supposed to
# catch each one is named beside it.
SABOTAGE = [
    ("threshold", "SPAM_MODEL_DATA.threshold += 1e-12;",
     "a decision cut out by a trillionth"),
    # The other half of the threshold. This one reports the right cut and
    # then applies 0.5 -- which is what a port that forgot the model carries
    # its own threshold would do, and what no comparison of the *number*
    # could see. It is caught, if it is caught, by a flipped verdict, which
    # is why check_threshold insists the corpus has a message in the band.
    #
    # The first read is honest because that is the one spam.js makes as it
    # defines itself, and it is the value SpamModel.threshold then reports.
    ("applied-cut",
     "(function () { var real = SPAM_MODEL_DATA.threshold, first = true;"
     " Object.defineProperty(SPAM_MODEL_DATA, 'threshold', { get: function () {"
     " if (first) { first = false; return real; } return 0.5; } }); }());",
     "the right cut reported, 0.5 applied"),
    ("idf", "SPAM_MODEL_DATA.idf[7] += 1e-6;",
     "one idf weight out by a millionth"),
    ("logprob", "if (SPAM_MODEL_DATA.logProb) "
                "{ SPAM_MODEL_DATA.logProb[1][11] += 1e-6; } "
                "else { SPAM_MODEL_DATA.coef[11] += 1e-6; }",
     "one fitted weight out by a millionth"),
    ("vocabulary", "SPAM_MODEL_DATA.terms[3] += 'x';",
     "one vocabulary term misspelt"),
    ("shoutRatio", "SPAM_MODEL_DATA.shoutRatio = 0.9;",
     "a different idea of when capitals count"),
]


def prove(instance, written, reference):
    """Break the bundle six ways and require the checks to catch each."""
    print("proving the comparison is not vacuous...")
    survived = []
    for name, injection, description in SABOTAGE:
        broken = dict(written)
        broken["js/model-data.js"] = (written["js/model-data.js"]
                                      + "\n/* --prove */ " + injection + "\n")
        try:
            run_checks(instance, broken, reference, loud=False)
        except SystemExit as refusal:
            lines = [line.strip() for line in str(refusal).splitlines()
                     if line.strip()]
            print("  %-12s %-36s caught" % (name, description))
            print("               %s" % (lines[0][:104]))
            if len(lines) > 2:
                print("               %s" % (lines[2][:104]))
            continue
        survived.append("%s (%s)" % (name, description))

    if survived:
        stop("these deliberate breakages went undetected, so the checks above "
             "are not covering what they claim to:\n  " + "\n  ".join(survived))
    print("  all %d caught -- the checks fail when the port is wrong\n"
          % len(SABOTAGE))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def page_html(template, payload):
    text = template

    text = replace_once(text, "<title>Spam Classifier</title>",
                        "<title>Spam Classifier \u2014 browser build</title>",
                        "retitle the page")

    text = replace_once(
        text,
        "<link rel=\"stylesheet\" href=\"{{ url_for('static', "
        "filename='css/style.css') }}\">",
        '<link rel="stylesheet" href="css/style.css">\n'
        '<link rel="stylesheet" href="css/static.css">',
        "point the stylesheet at the copied CSS")

    text = replace_once(
        text,
        "<h1>Spam Classifier<em>TF-IDF + scikit-learn</em></h1>",
        "<h1>Spam Classifier<em>browser build \u2014 exported "
        "TF-IDF model</em></h1>",
        "relabel the heading")

    text = replace_once(text, "  </header>\n", "  </header>\n\n" + banner(payload),
                        "add the build banner under the header")

    text = replace_once(
        text,
        "  <footer>Trained on the CSV corpus in <code>data/</code>",
        "  <footer>Inference only: this page carries the numbers a training "
        "run produced and multiplies\n    them out. Trained on the CSV corpus "
        "in <code>data/</code>",
        "say in the footer that this build does not train")

    scripts = "\n".join('<script src="%s"></script>' % name
                        for name in SCRIPT_ORDER)
    text = replace_once(
        text,
        "<script src=\"{{ url_for('static', filename='js/app.js') }}\"></script>",
        "<!-- Order matters. model-data.js is read by spam.js as it defines\n"
        "     itself; static-api.js installs the fetch shim before app.js\n"
        "     runs and immediately calls it; static-ui.js corrects a control\n"
        "     app.js binds and goes last. app.js itself is the Flask app's\n"
        "     file, copied byte for byte -- see tools/build_static.py. -->\n"
        + scripts,
        "rewrite the script tags")

    left = re.findall(r"\{\{.*?\}\}|\{%.*?%\}", text)
    if left:
        stop("unresolved Jinja left in the page: %r" % left)
    return text


def banner(payload):
    return """  <!-- Added by tools/build_static.py. The Flask app does not have this
       paragraph because the Flask app does not need it: it has scikit-learn
       behind it and can refit the model. This page cannot, and the two are
       otherwise identical, so the difference has to be said out loud. -->
  <div class="staticBanner">
    <b>This is the browser-only build.</b> There is no Python running. The
    fitted model &mdash; %s at a decision threshold of
    %.4f, over a %s-term TF-IDF vocabulary &mdash; was exported to
    <code>js/model-data.js</code>, and the scoring path from
    <code>spamlib.py</code> was ported to <code>js/spam.js</code>: normalise,
    vectorise, one sparse dot product, one comparison against the threshold.
    <b>It does inference only</b>; training needs scikit-learn and the corpus,
    so the Retrain button is disabled here. The port is not taken on trust:
    <code>tools/build_static.py</code> runs both implementations over every
    message in <code>data/</code> plus a list of deliberately awkward ones, and
    refuses to publish this page if a single verdict, probability, cleaned
    string or token weight disagrees.
    <a href="%s">Source, and the Flask app this is a copy of &rarr;</a>
  </div>
""" % (payload["chosen"], payload["threshold"],
       format(len(payload["terms"]), ","), REPO)


# ---------------------------------------------------------------------------
def prune(kept):
    """Delete anything left in docs/app/ from an older build.

    Without this a renamed file lives on and is still loaded, which is the one
    failure mode a generated directory has that a hand-written one does not.
    """
    if not os.path.isdir(OUT):
        return
    for here, _dirs, names in os.walk(OUT):
        for name in names:
            path = os.path.join(here, name)
            if os.path.relpath(path, OUT).replace("\\", "/") not in kept:
                os.remove(path)
                print("  removed stale %s" % os.path.relpath(path, OUT))


def collect(payload):
    """Everything the bundle is made of, as {path in docs/app: text}."""
    written = {"js/model-data.js": data_file(payload)}
    for name in SHARED_JS:
        written["js/" + name] = (
            BANNER % ("static/js/" + name)
            + read(os.path.join(ROOT, "static", "js", name)))
    for name in STATIC_JS:
        written["js/" + name] = read(os.path.join(SRC, "js", name))
    written["css/style.css"] = (
        BANNER % "static/css/style.css"
        + read(os.path.join(ROOT, "static", "css", "style.css")))
    written["css/static.css"] = read(os.path.join(SRC, "css", "static.css"))
    written["index.html"] = page_html(
        read(os.path.join(ROOT, "templates", "index.html")), payload)
    return written


def sizes(written):
    """Raw and gzipped bytes per file and in total.

    Gzipped as well as raw because GitHub Pages serves gzip and the raw number
    of a file that is nine tenths decimal digits is misleading by a factor of
    three.
    """
    rows = []
    for name, body in sorted(written.items()):
        raw = body.encode("utf-8")
        rows.append((name, len(raw), len(gzip.compress(raw, 9))))
    return rows


def main(argv):
    proving = "--prove" in argv[1:]
    for flag in argv[1:]:
        if flag != "--prove":
            stop("unknown option %r. The only one is --prove." % flag)

    model, vectorizer = load_model()
    payload = collect_model(model, vectorizer)
    payload["metrics"], payload["datasetSize"] = model_metrics(model, vectorizer)
    print("exporting %s at threshold %.4f, %d tf-idf terms"
          % (payload["chosen"], payload["threshold"], len(payload["terms"])))

    written = collect(payload)
    messages = corpus()
    print("checking the port against the Python it came from, over %d "
          "messages..." % len(messages))
    reference = python_reference(messages, model, vectorizer)

    with chromium() as instance:
        run_checks(instance, written, reference)
        if proving:
            print()
            prove(instance, written, reference)

    for name, body in sorted(written.items()):
        write(os.path.join(OUT, name.replace("/", os.sep)), body)
    prune(set(written))

    rows = sizes(written)
    print("wrote %d files to docs/app" % len(rows))
    for name, raw, packed in rows:
        print("  %-22s %7.1f KB  %6.1f KB gzipped"
              % (name, raw / 1024.0, packed / 1024.0))
    print("  %-22s %7.1f KB  %6.1f KB gzipped"
          % ("total", sum(r[1] for r in rows) / 1024.0,
             sum(r[2] for r in rows) / 1024.0))
    print("  serve it:  python -m http.server -d docs 8000  ->  "
          "http://127.0.0.1:8000/app/")


if __name__ == "__main__":
    main(sys.argv)
