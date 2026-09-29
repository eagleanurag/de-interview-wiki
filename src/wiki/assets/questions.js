/*
 * Filtering for the interview question browser.
 *
 * Every question card is already present in the server-generated HTML.
 * This script only toggles the `hidden` attribute on those cards, so
 * the page is complete without JavaScript and no content is ever
 * constructed from strings.
 */
(function () {
  "use strict";

  var query = document.getElementById("question-query");
  var topic = document.getElementById("question-topic");
  var difficulty = document.getElementById("question-difficulty");
  var type = document.getElementById("question-type");
  var reset = document.getElementById("question-reset");
  var status = document.getElementById("question-status");
  var list = document.getElementById("question-list");

  if (!list) {
    return;
  }

  var cards = Array.prototype.slice.call(
    list.querySelectorAll(".question-card")
  );

  cards.forEach(function (card) {
    var raw = card.getAttribute("data-topics") || "[]";
    var parsed = [];

    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      parsed = [];
    }

    card._topics = Array.isArray(parsed) ? parsed : [];
    card._haystack = (
      (card.getAttribute("data-search") || "") +
      " " +
      card._topics.join(" ")
    ).toLowerCase();
  });

  function apply() {
    var terms = (query && query.value ? query.value : "")
      .toLowerCase()
      .trim();
    var tokens = terms.length > 0 ? terms.split(/\s+/) : [];
    var wantedTopic = topic ? topic.value : "";
    var wantedDifficulty = difficulty ? difficulty.value : "";
    var wantedType = type ? type.value : "";
    var shown = 0;

    cards.forEach(function (card) {
      var visible = true;

      if (wantedTopic && card._topics.indexOf(wantedTopic) === -1) {
        visible = false;
      }

      if (
        visible &&
        wantedDifficulty &&
        card.getAttribute("data-difficulty") !== wantedDifficulty
      ) {
        visible = false;
      }

      if (
        visible &&
        wantedType &&
        card.getAttribute("data-type") !== wantedType
      ) {
        visible = false;
      }

      if (visible && tokens.length > 0) {
        visible = tokens.every(function (token) {
          return card._haystack.indexOf(token) !== -1;
        });
      }

      card.hidden = !visible;

      if (visible) {
        shown += 1;
      }
    });

    if (status) {
      status.textContent =
        shown +
        (shown === 1 ? " question shown" : " questions shown") +
        " of " +
        cards.length +
        ".";
    }
  }

  function resetAll() {
    if (query) {
      query.value = "";
    }

    [topic, difficulty, type].forEach(function (element) {
      if (element) {
        element.value = "";
      }
    });

    apply();
  }

  [query, topic, difficulty, type].forEach(function (element) {
    if (!element) {
      return;
    }

    element.addEventListener("input", apply);
    element.addEventListener("change", apply);
  });

  if (reset) {
    reset.addEventListener("click", resetAll);
  }

  apply();
})();
