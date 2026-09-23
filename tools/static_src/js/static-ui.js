/* ---------------------------------------------------------------------------
   static-ui.js — the one control this build cannot honour
   ---------------------------------------------------------------------------
   Loaded last, after app.js, because it corrects a control app.js binds.
   Nothing in static/js/app.js is edited to make this build work: it is copied
   byte for byte by tools/build_static.py, and the moment it is forked the two
   copies start drifting and the published one is the one nobody runs the
   tests against.

   The Retrain button is the whole of it. In the Flask app it refits three
   estimators under five-fold cross-validation, sweeps the decision threshold
   over the out-of-fold scores and saves the winner; here there is no
   scikit-learn and no corpus, only the numbers the winner ended up with. The
   button is therefore left in the markup -- app.js binds it at load and a
   build that deleted it would throw before the page drew -- but disabled,
   relabelled, and given the reason underneath, so nobody presses it twice
   wondering why nothing happened.

   The element stays rather than being hidden for the same reason: a control
   that is visibly unavailable is information, and one that quietly vanished
   between two otherwise identical pages is a mystery.
   --------------------------------------------------------------------------- */

(function () {
  'use strict';

  function el(id) { return document.getElementById(id); }

  var button = el('retrainBtn');
  if (button) {
    button.disabled = true;
    button.textContent = 'Retrain (needs the Flask app)';
    button.title = 'Training runs scikit-learn; this build only scores.';
  }

  /* Written after app.js's loadModel() has resolved, so it is not overwritten
     by the note renderModel puts there. renderModel is called from a promise,
     so a microtask is not enough -- this waits for the shim's reply to have
     been rendered. */
  function annotate() {
    var note = el('mNote');
    if (!note) { return; }
    note.textContent = note.textContent +
      ' Vocabulary: ' + SpamModel.vocabularySize.toLocaleString() +
      ' tf-idf terms, exported from the fitted model; decision threshold ' +
      SpamModel.threshold.toFixed(4) + '.';
  }

  if (document.readyState === 'complete') { setTimeout(annotate, 0); }
  else { window.addEventListener('load', function () { setTimeout(annotate, 0); }); }
}());
