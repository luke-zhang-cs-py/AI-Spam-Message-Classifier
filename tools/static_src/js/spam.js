/* ---------------------------------------------------------------------------
   spam.js — the scoring half of spamlib.py, ported to the browser
   ---------------------------------------------------------------------------
   This is inference only, and deliberately so. Training is scikit-learn's
   cross-validated model comparison and threshold sweep over the whole corpus;
   nothing here fits anything. What it does is the one path a reader of the
   page actually exercises:

       spamlib.clean_text            ->  cleanText
       TfidfVectorizer.transform     ->  vectorize
       MultinomialNB.predict_proba   ->  score          (via SPAM_MODEL_DATA)
       Thresholded.predict           ->  classify
       app.token_weights             ->  tokenWeights

   The numbers it multiplies -- the vocabulary, the idf vector, the fitted
   log-probabilities, the class priors and the chosen threshold -- are not
   re-derived here. They are exported from the fitted artifacts into
   js/model-data.js by tools/build_static.py, which then runs this file
   against the real Python over the whole corpus and refuses to publish the
   bundle if a single verdict, probability, cleaned string or token weight
   disagrees. So a mistake in this file is a build failure, not a wrong answer
   on a page nobody is testing.

   Three places where the port cannot just be transcribed, because Python's
   regular expressions and JavaScript's do not mean the same thing by the same
   characters:

     * `\w` in Python (on str patterns) is "isalnum() or underscore", which is
       every Unicode letter and number. JavaScript's `\w` is [A-Za-z0-9_] and
       nothing else. Spelled out as [\p{L}\p{N}_] below -- verified against
       CPython character by character.
     * `\d` likewise: Python matches every Unicode decimal, JavaScript matches
       0-9. Spelled out as \p{Nd}.
     * `\s` differs in *both* directions. Python counts U+001C-U+001F (the
       file/group/record/unit separators) and U+0085 as whitespace and
       JavaScript does not; JavaScript counts U+FEFF and Python does not. A
       message pasted out of a spreadsheet can contain the first kind.

   and therefore `\b`, which is defined in terms of `\w`, has to be written
   out as a pair of lookarounds rather than used directly. That needs
   lookbehind: Chrome 62+, Firefox 78+, Safari 16.4+.
   --------------------------------------------------------------------------- */

'use strict';

var SpamModel = (function () {

  var DATA = SPAM_MODEL_DATA;

  /* ------------------------------------------------------- the classes */
  var W  = '[\\p{L}\\p{N}_]';                              /* Python's \w  */
  var SP = '\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680' +
           '\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
  var S  = '[' + SP + ']';                                 /* Python's \s  */
  var NS = '[^' + SP + ']';                                /* Python's \S  */
  var ND = '\\p{Nd}';                                      /* Python's \d  */
  var B  = '(?:(?<=' + W + ')(?!' + W + ')|(?<!' + W + ')(?=' + W + '))';

  function re(source, flags) { return new RegExp(source, flags + 'u'); }

  /* spamlib._URL through spamlib._SPACE, in the same order they are applied.
     The `i` on the first one is spamlib's re.I; it is redundant because
     clean_text lowercases first, and it is kept so the two read alike. */
  var URL_RE   = re('(?:https?://|www\\.)' + NS + '+|' +
                    B + NS + '+\\.(?:com|net|org|co|io|info|ly)' + B +
                    '/?' + NS + '*', 'gi');
  var MONEY_RE = re('[£$€]' + S + '?' + ND + '[' + ND + ',.]*', 'g');
  var PHONE_RE = re(B + '0' + ND + '{8,10}' + B, 'g');
  var SHORT_RE = re(B + ND + '{4,6}' + B, 'g');
  var DIGIT_RE = re(B + ND + '[' + ND + ',.]*' + B, 'g');
  var SPACE_RE = re(S + '+', 'g');
  var STRIP_RE = re('^' + S + '+|' + S + '+$', 'g');       /* str.strip()  */

  /* string.punctuation, exactly: U+0021-2F, U+003A-40, U+005B-60, U+007B-7E.
     Note that the underscore is in it, so `__init__` cleans to `init`. */
  var PUNCT_RE = /[!-\/:-@\[-`{-~]/g;

  var LETTER = re('\\p{L}', '');        /* str.isalpha() on one character */
  var UPPER  = re('\\p{Lu}', '');       /* str.isupper() on one letter    */

  /* sklearn's default token_pattern, (?u)\b\w\w+\b, is exactly "maximal runs
     of word characters, two or more long" -- a run of one cannot match it at
     all, and a longer run is consumed whole because \w+ is greedy and the
     trailing \b then holds. So no \b is needed here. */
  var TOKEN_RE = re(W + '{2,}', 'g');

  /* ------------------------------------------------------------ rounding */
  /* Python's round(x, 4). Not x.toFixed(4): ECMAScript rounds a tie away from
     zero and Python rounds it to even, and both do it on the *exact* binary
     value of x rather than on a decimal approximation of it. The page shows
     these numbers, so the build compares them exactly, so a tie has to land
     the same way on both sides. Exact ties are rare but reachable -- 0.15625
     is one -- and "rare" is not a property you can ship. */
  var BITS = new DataView(new ArrayBuffer(8));

  function round4(x) {
    if (!Number.isFinite(x)) { return x; }
    if (x < 0) { return -round4(-x); }
    if (x === 0) { return x; }
    BITS.setFloat64(0, x);
    var word = BITS.getBigUint64(0);
    var exponent = Number((word >> 52n) & 0x7ffn);
    var mantissa = word & 0xfffffffffffffn;
    if (exponent === 0) { exponent = 1; } else { mantissa |= 0x10000000000000n; }
    /* x is exactly mantissa * 2^(exponent-1075). */
    var shift = exponent - 1075;
    var numerator = mantissa * 10000n;
    var denominator = 1n;
    if (shift >= 0) { numerator <<= BigInt(shift); }
    else { denominator = 1n << BigInt(-shift); }
    var whole = numerator / denominator;
    var rest = (numerator % denominator) * 2n;
    if (rest > denominator || (rest === denominator && (whole & 1n) === 1n)) {
      whole += 1n;
    }
    return Number(whole) / 10000;
  }

  /* ------------------------------------------------------- clean_text */
  function cleanText(text) {
    text = (text === null || text === undefined) ? '' : String(text);

    /* Counted over code points, as Python iterates a str -- not over UTF-16
       units, which would count an emoji twice and neither half as a letter. */
    var letters = 0;
    var shouted = 0;
    for (var ch of text) {
      if (LETTER.test(ch)) {
        letters += 1;
        if (UPPER.test(ch)) { shouted += 1; }
      }
    }
    var shouting = letters > 0 && (shouted / letters) > DATA.shoutRatio;

    var out = text.toLowerCase();
    out = out.replace(URL_RE, ' urltoken ');
    out = out.replace(MONEY_RE, ' moneytoken ');
    out = out.replace(PHONE_RE, ' phonetoken ');
    out = out.replace(SHORT_RE, ' shortcodetoken ');
    out = out.replace(DIGIT_RE, ' numtoken ');
    out = out.replace(PUNCT_RE, '');
    out = out.replace(SPACE_RE, ' ').replace(STRIP_RE, '');

    return shouting ? out + ' allcapstoken' : out;
  }

  /* ------------------------------------------------- the fitted numbers */
  var TERMS = DATA.terms;
  var IDF = DATA.idf;

  /* app.token_weights asks the estimator for coef_ first and falls back to
     feature_log_prob_, because the two store the same linear weight
     differently. The build records which branch the fitted model takes, so
     this does not have to guess -- and the difference of the two log-prob
     rows is computed here rather than shipped, because it is exactly that:
     a subtraction of two numbers already in the bundle. */
  var WEIGHT = DATA.kind === 'linear'
    ? DATA.coef
    : DATA.logProb[1].map(function (v, i) { return v - DATA.logProb[0][i]; });

  var INDEX = new Map();
  for (var t = 0; t < TERMS.length; t += 1) { INDEX.set(TERMS[t], t); }

  /* --------------------------------------- TfidfVectorizer.transform */
  function analyze(doc) {
    /* CountVectorizer's analyzer: preprocess (lowercase), tokenize, then
       _word_ngrams for ngram_range (1, 2) -- the unigrams in order, then
       every adjacent pair joined by a space. */
    var unigrams = doc.toLowerCase().match(TOKEN_RE) || [];
    var grams = unigrams.slice();
    for (var i = 0; i + 1 < unigrams.length; i += 1) {
      grams.push(unigrams[i] + ' ' + unigrams[i + 1]);
    }
    return grams;
  }

  function vectorize(message) {
    var counts = new Map();
    analyze(cleanText(message)).forEach(function (gram) {
      var col = INDEX.get(gram);
      if (col !== undefined) { counts.set(col, (counts.get(col) || 0) + 1); }
    });

    /* Ascending column order, because that is the order scipy stores a CSR
       row in after sort_indices() and therefore the order every sum below is
       accumulated in. Floating-point addition is not associative, so this is
       not cosmetic: it is what makes the totals bit-for-bit the same. */
    var cols = Array.from(counts.keys()).sort(function (a, b) { return a - b; });
    var data = new Array(cols.length);
    var i;

    /* sublinear_tf, then idf, then l2 -- TfidfTransformer's order. */
    for (i = 0; i < cols.length; i += 1) {
      data[i] = (Math.log(counts.get(cols[i])) + 1) * IDF[cols[i]];
    }
    var square = 0;
    for (i = 0; i < data.length; i += 1) { square += data[i] * data[i]; }
    if (square !== 0) {
      var norm = Math.sqrt(square);
      for (i = 0; i < data.length; i += 1) { data[i] /= norm; }
    }
    return { cols: cols, data: data };
  }

  /* ------------------------------------------------------ predict_proba */
  function softmax(a, b) {
    var top = a > b ? a : b;
    var ea = Math.exp(a - top);
    var eb = Math.exp(b - top);
    var sum = ea + eb;
    return [ea / sum, eb / sum];
  }

  function score(vec) {
    var i;
    var col;
    var value;

    if (DATA.kind === 'linear') {
      /* LogisticRegression: decision_function, then softmax over [-d, d],
         which is what sklearn's binary predict_proba does. */
      var decision = DATA.intercept;
      for (i = 0; i < vec.cols.length; i += 1) {
        decision += vec.data[i] * DATA.coef[vec.cols[i]];
      }
      return softmax(-decision, decision);
    }

    /* MultinomialNB / ComplementNB: X . feature_log_prob_.T, the prior added
       afterwards, then exp(jll - logsumexp(jll)). ComplementNB does not add
       the prior in the two-class case, which is the one bit of the two that
       differs; the build records which. */
    var ham = 0;
    var spam = 0;
    for (i = 0; i < vec.cols.length; i += 1) {
      col = vec.cols[i];
      value = vec.data[i];
      ham += value * DATA.logProb[0][col];
      spam += value * DATA.logProb[1][col];
    }
    if (DATA.addPrior) {
      ham += DATA.logPrior[0];
      spam += DATA.logPrior[1];
    }
    return softmax(ham, spam);
  }

  /* ------------------------------------------------- app.token_weights */
  function tokenWeights(vec, limit) {
    if (!vec.cols.length) { return []; }
    var out = [];
    for (var i = 0; i < vec.cols.length; i += 1) {
      var col = vec.cols[i];
      out.push({ token: TERMS[col],
                 weight: round4(vec.data[i] * WEIGHT[col]),
                 tfidf: round4(vec.data[i]) });
    }
    /* By the magnitude of the *rounded* weight, which is what app.py sorts
       on, and stably, which is what Python's sort and V8's both are. */
    out.sort(function (a, b) { return Math.abs(b.weight) - Math.abs(a.weight); });
    return out.slice(0, limit === undefined ? 12 : limit);
  }

  /* ------------------------------------------------------- app.classify */
  function classify(message) {
    var vec = vectorize(message);
    var proba = score(vec);
    var spam = proba[1] >= DATA.threshold;
    return {
      message: message,
      cleaned: cleanText(message),
      label: spam ? 'spam' : 'ham',
      /* The confidence in the *verdict*, not P(spam) -- app.py reads the
         probability of the predicted class out by its position in classes_,
         so a confident "ham" reads 0.97 rather than 0.03. */
      confidence: round4(spam ? proba[1] : proba[0]),
      knownTokens: vec.cols.length,
      tokens: tokenWeights(vec)
    };
  }

  function spamProbability(message) {
    return score(vectorize(message))[1];
  }

  return {
    cleanText: cleanText,
    classify: classify,
    spamProbability: spamProbability,
    round4: round4,
    threshold: DATA.threshold,
    vocabularySize: TERMS.length
  };
}());
