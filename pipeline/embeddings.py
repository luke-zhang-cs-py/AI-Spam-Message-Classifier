"""
embeddings.py
--------------
The optional sentence-embedding backend: a pretrained transformer used as a
feature extractor beside the tf-idf columns.

Why it is optional, in numbers
==============================
Measured on the 412-message corpus, under the same 5-fold split, the same
PRECISION_FLOOR and the same out-of-fold threshold sweep as `spamlib`:

    features            estimator          precision  recall     F1
    tf-idf              MultinomialNB          0.900    0.804   0.850   <- default
    embeddings          LogReg                 0.903    0.726   0.805
    embeddings          RBF SVM                0.913    0.765   0.833
    tf-idf + embeddings LogReg balanced        0.901    0.810   0.853

So the transformer buys about three thousandths of F1 over tf-idf alone,
which is noise at this sample size, and *on its own it is worse* -- 0.833
against 0.850. That is the expected result and worth stating plainly rather
than burying: MiniLM was trained on general web text and has never seen a
smishing corpus, while tf-idf learns this corpus's own vocabulary directly,
and `moneytoken`, `shortcodetoken` and "claim" are strong enough evidence
that a general-purpose sentence encoder dilutes it.

It is here anyway, off by default, for two reasons. The gap should widen as
the corpus grows, because embeddings generalise to paraphrases a lexical
model has never seen; and the comparison itself is worth being able to
re-run as new material lands, which needs the code to exist.

Nothing imports torch unless you ask
====================================
`sentence_transformers` pulls in torch, which is about two gigabytes. Paying
that to import the module that reads a CSV would be absurd, so the import
lives inside `load_encoder` and everything above it works out of the box:
`available()` answers with `importlib.util.find_spec`, which does not
execute the package.

This module deliberately imports nothing from `spamlib`. It is a leaf --
`spamlib.build_vectorizer` reaches down to it and hands in the tf-idf
vectoriser and the text cleaner, rather than this file reaching back up.
That keeps the dependency one-directional and means `import embeddings`
cannot drag the whole pipeline in.

The model is local, and that is a requirement not an accident
=============================================================
`MODEL_DIR` is a directory on disk, never a hub id. Partly because
huggingface.co is not reachable from every network that wants to run this
(the certificate interception that also stops urllib fetching the ECB feed
in a sibling project), but mostly because a filter that reads somebody's
text messages should not need the network to do it. Nothing in this file
sends a message anywhere.
"""

import importlib.util
import os

import numpy as np
from scipy.sparse import csr_matrix, hstack

# The project root, not this package: the encoder is downloaded into
# models/ beside the code. One dirname would look in pipeline/models
# and report the encoder as missing on a machine that has it.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# all-MiniLM-L6-v2: 6 layers, 384 dimensions, ~87 MB. Chosen for being the
# smallest sentence encoder that is not obviously worse than the big ones on
# short text, which is the only kind of text this project has.
DEFAULT_MODEL_DIR = os.path.join(_HERE, "models", "all-MiniLM-L6-v2")
MODEL_DIR_ENV = "SPAM_EMBEDDING_MODEL"

# What the shipped default produces. Not enforced -- point MODEL_DIR_ENV at a
# different encoder and its own width is used -- but recorded, because a
# model that silently loads with the wrong width is otherwise only visible as
# a strange score.
MINILM_DIMENSION = 384

# Encoding is the expensive half and batching is most of the difference. 64
# is comfortably inside memory for messages this short.
BATCH_SIZE = 64

# The files sentence-transformers needs before it will load a directory. A
# partial download -- interrupted curl, half-synced folder -- otherwise fails
# deep inside the library with a message about a missing key.
REQUIRED_FILES = ("config.json", "modules.json", "tokenizer.json")

_encoders = {}


def model_directory(model_dir=None):
    """Where the encoder is, in precedence order: argument, environment,
    the bundled default."""
    return model_dir or os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR


def missing_requirement(model_dir=None):
    """Why this backend cannot be used, as a sentence, or None if it can.

    A string rather than a bool because the two reasons need different
    fixes and the caller should be able to say which one applies. "install
    the extras" is unhelpful advice to somebody who has them installed and
    is missing the weights.
    """
    if importlib.util.find_spec("sentence_transformers") is None:
        return ("sentence-transformers is not installed; "
                "pip install -r requirements-embeddings.txt")

    directory = model_directory(model_dir)
    if not os.path.isdir(directory):
        return ("no encoder at %s; see README.md for the two curl commands "
                "that fetch it, or set %s" % (directory, MODEL_DIR_ENV))

    absent = [name for name in REQUIRED_FILES
              if not os.path.isfile(os.path.join(directory, name))]
    if absent:
        return ("%s is not a complete sentence-transformers model: missing %s"
                % (directory, ", ".join(absent)))
    return None


def available(model_dir=None):
    return missing_requirement(model_dir) is None


def load_encoder(model_dir=None):
    """The SentenceTransformer for `model_dir`, loaded once per process.

    Cached because loading is around a second and a half of reading 87 MB
    and building the graph, and both `fit_transform` and every later
    `transform` want the same object. Keyed by directory so pointing the
    environment variable somewhere else during a test does not hand back the
    previous model.

    Raises RuntimeError with the sentence from `missing_requirement` rather
    than letting an ImportError or a library-internal KeyError out, because
    "no module named torch" is a worse answer to "why is this off" than the
    command that turns it on.
    """
    problem = missing_requirement(model_dir)
    if problem:
        raise RuntimeError(problem)

    directory = model_directory(model_dir)
    if directory not in _encoders:
        from sentence_transformers import SentenceTransformer
        _encoders[directory] = SentenceTransformer(directory)
    return _encoders[directory]


def reset():
    """Forget the cached encoder.

    Every module cache in this family of projects has one of these --
    fxrates.reset, schedule.reset, app.reset -- for the same reason: without
    it, a test that points MODEL_DIR_ENV at a fixture leaves every later test
    holding the fixture's encoder.
    """
    _encoders.clear()


def encode(texts, model_dir=None):
    """Unit-norm embeddings for a list of raw messages, one row each.

    Normalised because every consumer here is a linear model or an RBF
    kernel, both of which want comparable magnitudes, and because it makes
    the dot product a cosine similarity if anything later wants one.
    """
    texts = list(texts)
    encoder = load_encoder(model_dir)
    if not texts:
        # encode([]) is not reliably shaped across versions, and the caller
        # needs a width to hstack against.
        width = getattr(encoder, "get_sentence_embedding_dimension",
                        lambda: MINILM_DIMENSION)()
        return np.zeros((0, int(width)), dtype="float32")
    vectors = encoder.encode(texts, batch_size=BATCH_SIZE,
                             show_progress_bar=False,
                             normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


class SemanticFeatures:
    """tf-idf columns with sentence-embedding columns stacked beside them.

    Stands in for a TfidfVectorizer everywhere `spamlib` uses one, with one
    deliberate difference announced by `wants_raw_text`: this takes the
    *unmodified* message, not the normalised one.

    That is not an oversight. `clean_text` exists to help a bag-of-words
    model -- collapsing a short code to `shortcodetoken` is valuable when
    every distinct number would otherwise be its own useless feature -- and
    is actively counterproductive for a model that reads the sentence, which
    would rather see "text WIN to 80086" than "text win to shortcodetoken".
    So each half gets the form of the text it wants: the lexical leg cleans
    internally, the semantic leg does not.

    The tf-idf columns come first and `lexical_width` says how many there
    are, which is what lets the web app explain a verdict in terms of words.
    There is no honest per-word story to tell about dimension 137 of a
    sentence embedding, so the explanation stops at the boundary rather than
    listing 384 meaningless rows.
    """

    # The flag `spamlib.wants_raw_text` reads. An attribute rather than a
    # type check so nothing has to import this module -- and importing this
    # module is exactly what the optional half is trying to avoid.
    wants_raw_text = True

    def __init__(self, lexical=None, cleaner=None, model_dir=None):
        self.lexical = lexical
        self.cleaner = cleaner
        self.model_dir = model_directory(model_dir)
        self.lexical_width = 0
        self.dimension = None
        self._encoder = None

    # -- the vectoriser interface -------------------------------------------
    def fit_transform(self, texts):
        return self._build(list(texts), fit=True)

    def transform(self, texts):
        return self._build(list(texts), fit=False)

    def _build(self, texts, fit):
        """The feature matrix, always as CSR.

        Always, including when there is no lexical half. It used to return
        the raw dense array in that case, which made one method have two
        return types and cost callers the sparse interface: `app.classify`
        asks the matrix for `.nnz`, and an ndarray does not have it, so an
        embeddings-only vectoriser reached the web app as an
        AttributeError. sklearn's estimators take either, so nothing wanted
        the dense form; it was just what `encode` happened to hand back.
        """
        embedded = encode(texts, self.model_dir)
        self.dimension = int(embedded.shape[1])
        if self.lexical is None:
            return csr_matrix(embedded)

        cleaned = self._cleaned(texts)
        lexical = (self.lexical.fit_transform(cleaned) if fit
                   else self.lexical.transform(cleaned))
        self.lexical_width = int(lexical.shape[1])
        return hstack([lexical, csr_matrix(embedded)]).tocsr()

    def _cleaned(self, texts):
        if self.cleaner is None:
            return texts
        return [self.cleaner(text) for text in texts]

    def get_feature_names_out(self):
        """Lexical names, then one placeholder per embedding dimension.

        The placeholders are there so the length matches the column count --
        `app.token_weights` indexes this array by column and would read off
        the end otherwise.
        """
        names = ([] if self.lexical is None
                 else list(self.lexical.get_feature_names_out()))
        width = self.dimension or 0
        return np.array(names + ["embedding[%d]" % i for i in range(width)],
                        dtype=object)

    # -- pickling ------------------------------------------------------------
    def __getstate__(self):
        """Everything except the encoder.

        The artifact on disk holds the model *directory*, not 87 MB of
        weights: joblib would happily serialise the torch modules and
        produce a vectorizer.joblib two orders of magnitude bigger than the
        model it was meant to describe, which would then only load on a
        machine with the same torch version. A path reloads anywhere the
        weights exist.
        """
        state = dict(self.__dict__)
        state["_encoder"] = None
        return state

    def __repr__(self):
        return ("SemanticFeatures(lexical=%d, embedding=%s, model=%r)"
                % (self.lexical_width, self.dimension,
                   os.path.basename(self.model_dir)))
