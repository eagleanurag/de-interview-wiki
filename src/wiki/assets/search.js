/*
 * Client-side search over assets/search-index.json.
 *
 * Result cards are built with DOM APIs and textContent rather than
 * innerHTML, so indexed content is never parsed as markup. The index
 * is generated from escaped-by-the-server data, but treating it as
 * untrusted here too keeps the page safe even if the index is edited.
 */
(function () {
  "use strict";

  var config = readConfig();
  var input = document.getElementById("search-input");
  var topicSelect = document.getElementById("search-topic");
  var status = document.getElementById("search-status");
  var results = document.getElementById("search-results");

  if (!input || !status || !results) {
    return;
  }

  var records = [];
  var loaded = false;
  var loadFailed = false;

  function readConfig() {
    var node = document.getElementById("wiki-config");
    var fallback = { indexUrl: "assets/search-index.json" };

    if (!node) {
      return fallback;
    }

    try {
      var parsed = JSON.parse(node.textContent || "{}");
      return {
        indexUrl: parsed.indexUrl || fallback.indexUrl
      };
    } catch (error) {
      return fallback;
    }
  }

  function text(value) {
    return value === null || value === undefined ? "" : String(value);
  }

  function haystack(record) {
    if (record._haystack === undefined) {
      var parts = [
        record.i,
        record.s,
        record.x,
        record.p,
        record.a,
        record.tp,
        record.sb,
        record.c,
        record.t,
        record.q
      ];

      record._haystack = flatten(parts).toLowerCase();
    }

    return record._haystack;
  }

  function flatten(value) {
    if (Array.isArray(value)) {
      return value.map(flatten).join(" ");
    }

    if (value === null || value === undefined) {
      return "";
    }

    if (typeof value === "object") {
      return Object.keys(value).map(function (key) {
        return flatten(value[key]);
      }).join(" ");
    }

    return String(value);
  }

  function load() {
    if (loaded || loadFailed) {
      return;
    }

    status.textContent = "Loading index...";

    fetch(config.indexUrl, { credentials: "same-origin" })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (payload) {
        records = Array.isArray(payload.records)
          ? payload.records
          : [];
        loaded = true;
        apply();
      })
      .catch(function () {
        loadFailed = true;
        status.textContent =
          "Search index could not be loaded. Use the topics or " +
          "questions pages instead.";
      });
  }

  function apply() {
    var terms = text(input.value).toLowerCase().trim();
    var tokens = terms.length > 0 ? terms.split(/\s+/) : [];
    var topic = topicSelect ? topicSelect.value : "";
    var matches = [];

    records.forEach(function (record) {
      if (topic && (record.tp || []).indexOf(topic) === -1) {
        return;
      }

      if (tokens.length === 0) {
        matches.push(record);
        return;
      }

      var hay = haystack(record);
      var ok = true;

      for (var i = 0; i < tokens.length; i += 1) {
        if (hay.indexOf(tokens[i]) === -1) {
          ok = false;
          break;
        }
      }

      if (ok) {
        matches.push(record);
      }
    });

    render(matches, terms, topic !== "");
  }

  function render(matches, terms, filteredByTopic) {
    while (results.firstChild) {
      results.removeChild(results.firstChild);
    }

    if (matches.length === 0) {
      var empty = document.createElement("div");
      empty.className = "empty-state";

      var heading = document.createElement("h2");
      heading.textContent = "No matches";

      var note = document.createElement("p");
      note.textContent = filteredByTopic
        ? "Nothing matches this topic with the current search."
        : "Try a broader term, or browse topics and questions.";

      empty.appendChild(heading);
      empty.appendChild(note);
      results.appendChild(empty);

      status.textContent =
        "No matches" + (terms ? ' for "' + terms + '"' : "");
      return;
    }

    matches.forEach(function (record) {
      results.appendChild(buildCard(record));
    });

    status.textContent =
      matches.length +
      (matches.length === 1 ? " post matches" : " posts match") +
      (terms ? ' "' + terms + '"' : "") +
      (filteredByTopic ? " in the selected topic" : "") +
      ".";
  }

  function buildCard(record) {
    var card = document.createElement("article");
    card.className = "card result-card";

    var title = document.createElement("h3");
    var link = document.createElement("a");
    link.className = "text-link";
    link.href = record.u;
    link.textContent = record.i;
    title.appendChild(link);
    card.appendChild(title);

    var metaBits = [record.p, record.d];

    if (record.a) {
      metaBits.push(record.a);
    }

    metaBits.push(record.n + " questions");
    metaBits.push((record.c || []).length + " concepts");

    var meta = document.createElement("p");
    meta.className = "result-meta";
    meta.textContent = metaBits.filter(Boolean).join(" \u00b7 ");
    card.appendChild(meta);

    if (record.s) {
      var summary = document.createElement("p");
      summary.className = "result-summary";
      summary.textContent = record.s;
      card.appendChild(summary);
    }

    if (record.tp && record.tp.length > 0) {
      var row = document.createElement("div");
      row.className = "badge-row";

      record.tp.slice(0, 8).forEach(function (topic) {
        var badge = document.createElement("span");
        badge.className = "badge badge-topic";
        badge.textContent = topic;
        row.appendChild(badge);
      });

      card.appendChild(row);
    }

    return card;
  }

  input.addEventListener("input", function () {
    if (loaded) {
      apply();
    }
  });

  if (topicSelect) {
    topicSelect.addEventListener("change", function () {
      if (loaded) {
        apply();
      }
    });
  }

  input.addEventListener("focus", load, { once: true });
  load();
})();
