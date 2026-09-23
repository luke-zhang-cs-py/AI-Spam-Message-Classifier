/* ---------------------------------------------------------------------------
   static-api.js — app.py's routes, answered inside the page
   ---------------------------------------------------------------------------
   static/js/app.js talks to the server in four places:

     POST /api/classify   one message  -> verdict, confidence, token weights
     POST /api/batch      many         -> one verdict per line, plus a summary
     GET  /api/model                   -> what is loaded, and how it scores
     POST /api/retrain                 -> fit again from data/

   This intercepts exactly those and answers the first three from spam.js and
   the numbers in model-data.js, so the *same* app.js runs in both builds. A
   fetch shim rather than a forked app.js: a second copy of the file with its
   fetch calls edited out is a fork that looks like a copy, and it rots the
   first time somebody fixes a bug in one of them. app.js is copied byte for
   byte into docs/app/ by tools/build_static.py; this file is the whole of
   what stands between it and a server.

   The one route that cannot be answered is /api/retrain, and it is not
   faked. Retraining is scikit-learn fitting three estimators under five-fold
   cross-validation and sweeping a threshold over the out-of-fold scores; a
   browser build that returned `{"ok": true}` and left the model alone would
   be lying about the only button on the page that changes anything. It
   answers 400 with the reason, and static-ui.js disables the button and says
   the same thing in the panel, so nobody has to press it to find out.

   The status codes and the wording of every error are app.py's, not
   invented, and tools/build_static.py replays the same requests against
   Flask's test client and against this file and compares both.

   Loaded *before* app.js, which calls loadModel() as it finishes.
   --------------------------------------------------------------------------- */

(function () {
  'use strict';

  var DATA = SPAM_MODEL_DATA;

  /* Enough of a Response for app.js's three call sites -- `.ok`, `.status`,
     `.json()`. Not a real Response: constructing one is possible, but then
     the shim would be claiming to be the platform's fetch rather than a
     stand-in for four known callers, and the first thing to use a fifth
     feature of it would fail a long way from here. */
  function reply(status, body) {
    return Promise.resolve({
      ok: status >= 200 && status < 300,
      status: status,
      statusText: status < 400 ? 'OK' : 'Static build',
      json: function () { return Promise.resolve(body); },
      text: function () { return Promise.resolve(JSON.stringify(body)); }
    });
  }

  /* Python's str.strip(), which strips Python's idea of whitespace and not
     JavaScript's -- the two differ over U+001C-U+001F, U+0085 and U+FEFF, and
     a message made only of those is blank to app.py and not to String.trim.
     A page that accepted a message the server refuses would then show a
     verdict for something the Flask build calls empty. */
  var STRIP_RE = new RegExp(
    '^[\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a' +
    '\\u2028\\u2029\\u202f\\u205f\\u3000]+|' +
    '[\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a' +
    '\\u2028\\u2029\\u202f\\u205f\\u3000]+$', 'gu');

  function strip(text) { return String(text).replace(STRIP_RE, ''); }

  /* app.py */
  var MAX_MESSAGE_CHARS = DATA.maxMessageChars;
  var MAX_BATCH = DATA.maxBatch;

  /* Python's format(n, ',') for the two caps in the error messages. Written
     out rather than left to toLocaleString, whose grouping depends on which
     ICU data the browser shipped with. */
  function commas(n) {
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /* ------------------------------------------------------- /api/classify */
  function apiClassify(body) {
    var message = body.message;
    if (typeof message !== 'string' || !strip(message)) {
      return reply(400, { ok: false, error: 'Type a message first.' });
    }
    message = strip(message);
    if (message.length > MAX_MESSAGE_CHARS) {
      return reply(400, { ok: false,
                          error: 'Messages are capped at ' +
                                 commas(MAX_MESSAGE_CHARS) + ' characters.' });
    }
    var out = SpamModel.classify(message);
    out.ok = true;
    return reply(200, out);
  }

  /* ---------------------------------------------------------- /api/batch */
  /* app.clean_batch, rule for rule and message for message. Each of these
     exists in app.py because something got through without it: a bare string
     was iterated character by character, an int raised inside the
     comprehension and came back a 500, and 200 uncapped messages is hundreds
     of megabytes of work in one request. */
  function cleanBatch(raw) {
    if (!Array.isArray(raw)) {
      throw new Error("'messages' must be a list of strings.");
    }
    var item;
    for (var i = 0; i < raw.length; i += 1) {
      item = raw[i];
      if (item !== null && item !== undefined && typeof item !== 'string') {
        throw new Error('Every message must be a string.');
      }
    }
    var messages = [];
    for (i = 0; i < raw.length; i += 1) {
      if (typeof raw[i] === 'string' && strip(raw[i])) {
        messages.push(strip(raw[i]));
      }
    }
    if (!messages.length) { throw new Error('No messages to classify.'); }
    if (messages.length > MAX_BATCH) {
      throw new Error('Cap is ' + MAX_BATCH + ' messages at a time.');
    }
    for (i = 0; i < messages.length; i += 1) {
      if (messages[i].length > MAX_MESSAGE_CHARS) {
        throw new Error('Messages are capped at ' + commas(MAX_MESSAGE_CHARS) +
                        ' characters.');
      }
    }
    return messages;
  }

  function apiBatch(body) {
    var messages;
    try {
      messages = cleanBatch(body.messages);
    } catch (refusal) {
      return reply(400, { ok: false, error: refusal.message });
    }
    var results = messages.map(function (m) { return SpamModel.classify(m); });
    var spam = results.filter(function (r) { return r.label === 'spam'; }).length;
    return reply(200, { ok: true, results: results,
                        summary: { total: results.length, spam: spam,
                                   ham: results.length - spam } });
  }

  /* ---------------------------------------------------------- /api/model */
  function apiModel() {
    return reply(200, { name: DATA.name, metrics: DATA.metrics,
                        datasetSize: DATA.datasetSize });
  }

  /* -------------------------------------------------------- /api/retrain */
  function apiRetrain() {
    return reply(400, { ok: false,
                        error: 'This is the browser build: it carries the ' +
                               'exported model and can only score messages ' +
                               'with it. Run the Flask app to retrain.' });
  }

  /* ------------------------------------------------------------- routing */
  /* Matched on the tail of the path, not on equality, so the bundle works
     served from a subdirectory (/AI-Spam-Message-Classifier/app/) as well as
     from the root and from a file:// double-click. */
  var ROUTES = [
    ['/api/classify', apiClassify],
    ['/api/batch', apiBatch],
    ['/api/model', apiModel],
    ['/api/retrain', apiRetrain]
  ];

  function parse(init) {
    if (!init || init.body === null || init.body === undefined) { return {}; }
    try { return JSON.parse(init.body); } catch (bad) { return {}; }
  }

  var passthrough = window.fetch ? window.fetch.bind(window) : null;

  window.fetch = function (input, init) {
    var url = String(input && input.url ? input.url : input).split('?')[0];
    for (var i = 0; i < ROUTES.length; i += 1) {
      if (url === ROUTES[i][0] || url.endsWith(ROUTES[i][0])) {
        return ROUTES[i][1](parse(init));
      }
    }
    if (passthrough) { return passthrough(input, init); }
    return Promise.reject(new TypeError('Failed to fetch: ' + url));
  };
}());
