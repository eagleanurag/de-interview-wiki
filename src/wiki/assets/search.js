/*
 * Client-side search over assets/search-index.json, plus a separate
 * assets/ocr-index.json for what the slide images say.
 *
 * The OCR index is fetched once, and only after a search has actually
 * run. It is much larger than the main index -- every transcription in
 * the archive rather than one row per post -- and most searches do not
 * need it, so making every keystroke pay for it would be the wrong
 * trade.
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
  var ocrRecords = [];
  var loaded = false;
  var loadFailed = false;
  var ocrRequested = false;

  function readConfig() {
    var node = document.getElementById("wiki-config");
    var fallback = {
      indexUrl: "assets/search-index.json",
      ocrIndexUrl: "assets/ocr-index.json"
    };

    if (!node) {
      return fallback;
    }

    try {
      var parsed = JSON.parse(node.textContent || "{}");
      return {
        indexUrl: parsed.indexUrl || fallback.indexUrl,
        ocrIndexUrl: parsed.ocrIndexUrl || fallback.ocrIndexUrl
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
        record.q,
        record.bc
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

  /*
   * Fetch the transcription index once, in the background, and re-run
   * the current search when it lands. Failure is silent: a site whose
   * slides were never transcribed has no OCR index to find, and saying
   * so would be a false error rather than a useful one.
   */
  function loadOcr() {
    if (ocrRequested) {
      return;
    }

    ocrRequested = true;

    if (!config.ocrIndexUrl) {
      return;
    }

    fetch(config.ocrIndexUrl, { credentials: "same-origin" })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (payload) {
        ocrRecords = Array.isArray(payload.records)
          ? payload.records
          : [];

        // Not before the main index has produced its first render: an
        // apply() over an empty record set would replace real results
        // with "No matches" and then have to take them back.
        if (loaded) {
          apply();
        }
      })
      .catch(function () {
        ocrRecords = [];
      });
  }

  function ocrHaystack(record) {
    if (record._ocrHaystack === undefined) {
      record._ocrHaystack = flatten([
        record.i,
        record.f,
        record.x,
        record.st,
        record.ql,
        record.m
      ]).toLowerCase();
    }

    return record._ocrHaystack;
  }

  function matchOcr(tokens) {
    var hits = [];

    for (var i = 0; i < ocrRecords.length; i += 1) {
      var record = ocrRecords[i];
      var hay = ocrHaystack(record);
      var ok = true;

      for (var t = 0; t < tokens.length; t += 1) {
        if (hay.indexOf(tokens[t]) === -1) {
          ok = false;
          break;
        }
      }

      if (ok) {
        hits.push(record);
      }
    }

    return hits;
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

    // A term that appears only inside a slide finds the slide, not the
    // post. Rendered in the same result list so a reader is not sent to
    // a second page, and capped so one word repeated across a hundred
    // slides does not bury the posts that discuss it in prose.
    var ocrHits = tokens.length === 0 ? [] : matchOcr(tokens).slice(0, 40);

    // A subject or subtopic result is what someone typing "broadcast
    // join" wanted; a post record that merely mentions it is a weaker
    // answer to the same query. Same-kind ordering is left to the
    // existing sort so nothing else about ranking changes.
    var RANK = { s: 0, b: 0, q: 1 };

    matches.sort(function (a, b) {
      var ra = RANK[a.k] === undefined ? 2 : RANK[a.k];
      var rb = RANK[b.k] === undefined ? 2 : RANK[b.k];

      return ra - rb;
    });

    render(matches, ocrHits, terms, topic !== "");
  }

  function render(matches, ocrHits, terms, filteredByTopic) {
    while (results.firstChild) {
      results.removeChild(results.firstChild);
    }

    if (matches.length === 0 && ocrHits.length === 0) {
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

    if (ocrHits.length > 0) {
      var heading2 = document.createElement("h2");
      heading2.className = "ocr-heading";
      heading2.textContent =
        "Text inside slide images (machine transcription)";

      results.appendChild(heading2);

      ocrHits.forEach(function (record) {
        results.appendChild(buildOcrCard(record));
      });
    }

    var parts = [];

    if (matches.length > 0) {
      parts.push(
        matches.length +
          (matches.length === 1 ? " post matches" : " posts match")
      );
    }

    if (ocrHits.length > 0) {
      parts.push(
        ocrHits.length +
          (ocrHits.length === 1 ? " slide matches" : " slides match")
      );
    }

    status.textContent =
      parts.join(", ") +
      (terms ? ' "' + terms + '"' : "") +
      (filteredByTopic ? " in the selected topic" : "") +
      ".";
  }

  /*
   * A transcription hit. The slide number and the file name are on the
   * card because a hit inside a picture is otherwise unexplainable: the
   * reader needs to know which slide to look at, and "OCR" alone does
   * not tell them whether to trust it.
   */
  function buildOcrCard(record) {
    var card = document.createElement("article");
    card.className = "card result-card ocr-card";

    var title = document.createElement("h3");

    var link = document.createElement("a");
    link.className = "text-link";
    link.href = record.u;
    link.textContent = record.i + " · slide " + record.s;

    title.appendChild(link);
    card.appendChild(title);

    var metaBits = [record.f];

    if (record.ql) {
      metaBits.push("transcription: " + record.ql);
    }

    if (record.st && record.st !== "AVAILABLE") {
      metaBits.push("status: " + record.st);
    }

    var meta = document.createElement("p");
    meta.className = "result-meta";
    meta.textContent = metaBits.filter(Boolean).join(" · ");

    card.appendChild(meta);

    var body = document.createElement("p");
    body.className = "result-summary";
    body.textContent = text(record.x).slice(0, 400);

    card.appendChild(body);

    var note = document.createElement("p");
    note.className = "muted result-note";
    note.textContent =
      "Machine transcription of the image. It has not been reviewed " +
      "by a person.";

    card.appendChild(note);

    return card;
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

    var KINDS = {
      s: "subject",
      b: "subtopic",
      q: "question",
      p: "post",
      t: "topic",
      c: "concept",
      x: "technology"
    };

    var metaBits = [KINDS[record.k] || record.p, record.d];

    if (record.bc) {
      metaBits.push(record.bc);
    }

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

  // Fetched after the main index has been applied, so the first result
  // set appears immediately and the slide text joins it a moment later
  // rather than the page sitting on "Loading index..." for both.
  window.setTimeout(loadOcr, 0);
})();
