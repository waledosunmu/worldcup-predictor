/* WC26 Predictor — client interactivity.
 * Fetches data.json once, then enhances whichever page it's on (detected by
 * the presence of container ids). Pages stay valid with JS off. */
(function () {
  "use strict";

  var CSS = getComputedStyle(document.documentElement);
  var MODEL = (CSS.getPropertyValue("--model") || "#2563eb").trim();
  var MARKET = (CSS.getPropertyValue("--market") || "#d97706").trim();
  var MUTED = (CSS.getPropertyValue("--muted") || "#6b7280").trim();
  var LINE = (CSS.getPropertyValue("--line") || "#e5e7eb").trim();

  var PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#0891b2",
    "#ca8a04", "#db2777", "#475569", "#ea580c", "#0d9488"];

  function pct(x) { return (x * 100).toFixed(x >= 0.1 ? 0 : 1) + "%"; }
  function pctSigned(x) { return (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%"; }
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // Run each enhancer in isolation: a runtime error in one must not blank the
  // rest of the page's interactivity.
  function guard(name, fn) {
    try { fn(); } catch (e) { console.error(name + " failed", e); }
  }

  fetch("data.json").then(function (r) { return r.json(); }).then(function (DATA) {
    var teamsByName = {};
    DATA.teams.forEach(function (t) { teamsByName[t.name] = t; });

    guard("table", function () { enhanceTable(); });
    guard("divergence", function () { enhanceDivergence(DATA); });
    guard("movers", function () { enhanceMovers(DATA); });
    guard("trackRecord", function () { enhanceTrackRecord(DATA); });
    guard("trends", function () { enhanceTrends(DATA); });
    guard("bracket", function () { enhanceBracket(DATA, teamsByName); });
    guard("modal", function () { setupModal(DATA, teamsByName); });
  }).catch(function (e) { console.error("data.json load failed", e); });

  /* ---------- B1: sortable/filterable team table ---------- */
  function enhanceTable() {
    var table = document.getElementById("team-table");
    if (!table) return;
    var tbody = table.tBodies[0];
    var ths = table.querySelectorAll("th.sortable");

    function rowVal(row, idx, key) {
      var cell = row.cells[idx];
      if (key === "team" || key === "group") return cell.textContent.trim().toLowerCase();
      var dv = cell.getAttribute("data-v");
      return dv != null ? parseFloat(dv) : 0;
    }

    var sortState = { key: null, dir: -1 };
    function sortBy(idx, th) {
      var key = th.getAttribute("data-key");
      var dir;
      if (sortState.key === key) dir = -sortState.dir;
      else dir = th.classList.contains("num") ? -1 : 1;
      sortState = { key: key, dir: dir };
      ths.forEach(function (h) { h.classList.remove("asc", "desc"); });
      th.classList.add(dir === 1 ? "asc" : "desc");
      var rows = Array.prototype.slice.call(tbody.rows);
      rows.sort(function (a, b) {
        var va = rowVal(a, idx, key), vb = rowVal(b, idx, key);
        if (va < vb) return -dir; if (va > vb) return dir; return 0;
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
    }
    ths.forEach(function (th, i) {
      th.addEventListener("click", function () { sortBy(i, th); });
    });

    var search = document.getElementById("team-search");
    var groupFilter = document.getElementById("group-filter");
    function applyFilter() {
      var q = (search && search.value || "").trim().toLowerCase();
      var g = groupFilter && groupFilter.value || "";
      Array.prototype.forEach.call(tbody.rows, function (row) {
        var name = row.cells[0].textContent.toLowerCase();
        var grp = row.getAttribute("data-group");
        var ok = (!q || name.indexOf(q) !== -1) && (!g || grp === g);
        row.style.display = ok ? "" : "none";
      });
    }
    if (search) search.addEventListener("input", applyFilter);
    if (groupFilter) groupFilter.addEventListener("change", applyFilter);

    // champion model/market toggle: emphasise the relevant column
    var toggle = document.getElementById("champ-toggle");
    var MODEL_COL = 3, MARKET_COL = 4;
    function setMode(mode) {
      var hi = mode === "market" ? MARKET_COL : MODEL_COL;
      var lo = mode === "market" ? MODEL_COL : MARKET_COL;
      Array.prototype.forEach.call(table.rows, function (row) {
        if (row.cells[hi]) row.cells[hi].style.opacity = "1";
        if (row.cells[lo]) row.cells[lo].style.opacity = ".45";
      });
      if (toggle) toggle.querySelectorAll("button").forEach(function (b) {
        b.classList.toggle("active", b.getAttribute("data-mode") === mode);
        b.classList.toggle("market", mode === "market");
      });
    }
    if (toggle) {
      toggle.querySelectorAll("button").forEach(function (b) {
        b.addEventListener("click", function () { setMode(b.getAttribute("data-mode")); });
      });
      setMode("model");
    }
  }

  /* ---------- B3: divergence diverging bar chart ---------- */
  function enhanceDivergence(DATA) {
    var host = document.getElementById("divergence");
    if (!host || !DATA.divergence.length || typeof Chart === "undefined") return;
    var rows = DATA.divergence.slice(0, 16);
    var canvas = el("canvas");
    canvas.height = Math.max(220, rows.length * 24);
    host.appendChild(canvas);
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: rows.map(function (d) { return d.team; }),
        datasets: [{
          label: "model − market",
          data: rows.map(function (d) { return d.diff * 100; }),
          backgroundColor: rows.map(function (d) {
            return d.diff >= 0 ? MODEL : MARKET;
          }),
        }],
      },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: true, text: "Where the model disagrees with the market (pp)" },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var d = rows[ctx.dataIndex];
                return "model " + pct(d.model) + " vs market " + pct(d.market) +
                  " (" + pctSigned(d.diff) + ")";
              },
              afterBody: function (items) {
                var t = teamLookup(DATA, rows[items[0].dataIndex].team);
                return t ? t.blurb : "";
              },
            },
          },
        },
        scales: { x: { title: { display: true, text: "model minus market (pp)" } } },
        onClick: function (evt, items) {
          if (items.length) openTeamModal(rows[items[0].index].team);
        },
      },
    });
  }
  function teamLookup(DATA, name) {
    for (var i = 0; i < DATA.teams.length; i++)
      if (DATA.teams[i].name === name) return DATA.teams[i];
    return null;
  }

  /* ---------- B6: movers ---------- */
  function enhanceMovers(DATA) {
    var host = document.getElementById("movers");
    if (!host) return;
    var box = el("div", "mcard");
    box.appendChild(el("h3", null, "Title-odds movers"));
    if (!DATA.movers.length) {
      box.appendChild(el("p", "muted", "Needs &ge;2 matchdays of history before movers appear."));
    } else {
      var list = el("div");
      DATA.movers.forEach(function (m) {
        var up = m.delta >= 0;
        var row = el("div", "advrow");
        row.innerHTML = '<a class="team-link" data-team="' + esc(m.team) + '">' +
          esc(m.team) + '</a> <span class="' + (up ? "delta-up" : "delta-down") +
          '">' + (up ? "&#9650; " : "&#9660; ") + pctSigned(m.delta) + '</span>' +
          ' <span class="muted">' + pct(m.from) + " &rarr; " + pct(m.to) + "</span>";
        list.appendChild(row);
      });
      box.appendChild(list);
    }
    host.appendChild(box);
  }

  /* ---------- B6: live track record ---------- */
  function enhanceTrackRecord(DATA) {
    var host = document.getElementById("track-record");
    if (!host) return;
    var box = el("div", "mcard");
    box.appendChild(el("h3", null, "Live track record (out-of-sample)"));
    var tr = DATA.trackRecord;
    if (!tr) {
      box.appendChild(el("p", "muted", "No completed matches with a locked prediction yet."));
      host.appendChild(box);
      return;
    }
    function fmt(x) { return x.toFixed(3); }
    var tbl = el("table");
    tbl.innerHTML = "<tr><th></th><th class='num'>Log loss</th><th class='num'>Brier</th>" +
      "<th class='num'>RPS</th></tr>" +
      "<tr><td><span class='badge model'>Model</span></td><td class='num'>" +
      fmt(tr.model.logloss) + "</td><td class='num'>" + fmt(tr.model.brier) +
      "</td><td class='num'>" + fmt(tr.model.rps) + "</td></tr>" +
      "<tr><td><span class='badge market'>Market</span></td><td class='num'>" +
      fmt(tr.market.logloss) + "</td><td class='num'>" + fmt(tr.market.brier) +
      "</td><td class='num'>" + fmt(tr.market.rps) + "</td></tr>";
    box.appendChild(tbl);
    var wins = tr.matches.filter(function (m) { return m.model.logloss < m.market.logloss; }).length;
    box.appendChild(el("p", "muted", "Over " + tr.n + " completed match" +
      (tr.n === 1 ? "" : "es") + " (lower is better). Model's per-match log loss beat the " +
      "market in <b>" + wins + " of " + tr.n + "</b>."));
    if (tr.n < 20) {
      box.appendChild(el("p", "muted caveat",
        "&#9888; Small sample (" + tr.n + ") — directional, not yet conclusive."));
    }
    box.appendChild(el("p", "muted small",
      "Per-match log loss — winner in bold, the lower (better) call highlighted."));
    var rows = "<tr><th>Match</th><th class='num'>Score</th><th class='num'>Model</th>" +
      "<th class='num'>Market</th></tr>";
    tr.matches.forEach(function (m) {
      var mWin = m.model.logloss < m.market.logloss;
      var kWin = m.market.logloss < m.model.logloss;
      var home = m.outcome === "home" ? "<b>" + esc(m.home) + "</b>" : esc(m.home);
      var away = m.outcome === "away" ? "<b>" + esc(m.away) + "</b>" : esc(m.away);
      var sc = m.score ? m.score[0] + "&#8211;" + m.score[1] : "&middot;";
      rows += "<tr><td class='mtch'>" + home + " <span class='vs'>v</span> " + away +
        "</td><td class='num sc'>" + sc + "</td><td class='num" + (mWin ? " win" : "") +
        "'>" + m.model.logloss.toFixed(2) + "</td><td class='num" + (kWin ? " win" : "") +
        "'>" + m.market.logloss.toFixed(2) + "</td></tr>";
    });
    var mt = el("table", "tr-matches");
    mt.innerHTML = rows;
    box.appendChild(mt);
    host.appendChild(box);
  }

  /* ---------- B2: championship trend chart ---------- */
  function enhanceTrends(DATA) {
    var canvas = document.getElementById("trend-chart");
    if (!canvas || typeof Chart === "undefined") return;
    var hist = DATA.history;
    var labels = hist.map(function (h) { return h.as_of; });
    var metricSel = document.getElementById("trend-metric");
    var teamsHost = document.getElementById("trend-teams");
    var toggle = document.getElementById("trend-toggle");
    var caption = document.getElementById("trend-caption");
    var mode = "model";

    var top = DATA.teams.slice(0, 6).map(function (t) { return t.name; });
    var selected = {};
    DATA.teams.slice(0, 16).forEach(function (t) {
      var lab = el("label");
      var cb = el("input");
      cb.type = "checkbox"; cb.value = t.name;
      cb.checked = top.indexOf(t.name) !== -1;
      selected[t.name] = cb.checked;
      cb.addEventListener("change", function () { selected[t.name] = cb.checked; render(); });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(" " + t.name));
      teamsHost.appendChild(lab);
    });

    var chart = null;
    function metricKey() { return metricSel ? metricSel.value : "model_champion"; }
    function marketAvailable() { return metricKey() === "model_champion"; }
    function isPct(key) { return key !== "elo"; }

    function seriesFor(team, key) {
      return hist.map(function (h) {
        var bag = h[key] || {};
        var v = bag[team];
        return v == null ? null : (isPct(key) ? v * 100 : v);
      });
    }
    function render() {
      var key = metricKey();
      var useMarket = mode === "market" && marketAvailable();
      var sourceKey = useMarket ? "market_champion" : key;
      var picks = Object.keys(selected).filter(function (t) { return selected[t]; });
      var ds = picks.map(function (team, i) {
        var color = PALETTE[i % PALETTE.length];  // distinct per team in both modes
        return {
          label: team, data: seriesFor(team, sourceKey),
          borderColor: color, backgroundColor: color,
          borderWidth: 2, tension: 0.2,
          borderDash: useMarket ? [6, 4] : [],  // dashed = market, solid = model
        };
      });
      if (chart) chart.destroy();
      chart = new Chart(canvas, {
        type: "line",
        data: { labels: labels, datasets: ds },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: "bottom" } },
          scales: {
            y: {
              title: { display: true, text: isPct(sourceKey) ? "probability (%)" : "Elo" },
              beginAtZero: isPct(sourceKey),
            },
          },
        },
      });
      if (caption) {
        var note = DATA.meta.n_history_points + " snapshot" +
          (DATA.meta.n_history_points === 1 ? "" : "s") + " so far";
        if (mode === "market" && !marketAvailable())
          note += " — market series only available for the champion metric, showing model.";
        caption.textContent = note + ".";
      }
    }
    if (metricSel) metricSel.addEventListener("change", render);
    if (toggle) toggle.querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        mode = b.getAttribute("data-mode");
        toggle.querySelectorAll("button").forEach(function (x) {
          x.classList.toggle("active", x === b);
          x.classList.toggle("market", mode === "market");
        });
        render();
      });
    });
    render();
  }

  /* ---------- B4: interactive bracket ---------- */
  function enhanceBracket(DATA, teamsByName) {
    var host = document.getElementById("bracket");
    if (!host) return;
    var rounds = [
      { key: "reach_r32", title: "Round of 32", n: 32 },
      { key: "reach_r16", title: "Round of 16", n: 24 },
      { key: "reach_qf", title: "Quarter-finals", n: 16 },
      { key: "reach_sf", title: "Semi-finals", n: 12 },
      { key: "reach_final", title: "Final", n: 10 },
      { key: "model_champion", title: "Champion", n: 8 },
    ];
    var cols = el("div", "bracket-cols");
    rounds.forEach(function (rd) {
      var col = el("div", "bcol");
      var sorted = DATA.teams.slice().sort(function (a, b) { return b[rd.key] - a[rd.key]; })
        .filter(function (t) { return t[rd.key] > 0.001; });
      var top = sorted.slice(0, 8);
      var h = el("h4");
      h.textContent = rd.title;
      h.title = "Top: " + top.slice(0, 5).map(function (t) {
        return t.name + " " + pct(t[rd.key]);
      }).join(", ");
      col.appendChild(h);
      sorted.slice(0, rd.n).forEach(function (t) {
        var p = t[rd.key];
        var cell = el("div", "bteam");
        cell.setAttribute("data-team", t.name);
        var shade = 0.10 + 0.80 * Math.min(1, p);
        cell.style.background = "rgba(37,99,235," + shade.toFixed(2) + ")";
        // dark text on light fills, white on saturated fills — keeps the % legible
        cell.style.color = p >= 0.5 ? "#fff" : "var(--ink)";
        cell.style.borderColor = "transparent";
        cell.innerHTML = "<b>" + esc(t.name) + "</b><span class='p'>" + pct(p) + "</span>";
        col.appendChild(cell);
      });
      cols.appendChild(col);
    });
    host.appendChild(cols);
  }

  /* ---------- B5: per-team modal ---------- */
  var MODAL_DATA = null, MODAL_TEAMS = null, modalSpark = null;
  function setupModal(DATA, teamsByName) {
    MODAL_DATA = DATA; MODAL_TEAMS = teamsByName;
    var modal = document.getElementById("team-modal");
    if (!modal) return;
    document.addEventListener("click", function (e) {
      var el2 = e.target.closest("[data-team]");
      if (el2) { e.preventDefault(); openTeamModal(el2.getAttribute("data-team")); }
    });
    modal.addEventListener("click", function (e) {
      if (e.target === modal || e.target.closest(".close")) closeModal();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeModal();
    });
  }
  function closeModal() {
    var modal = document.getElementById("team-modal");
    if (modal) modal.classList.remove("open");
    if (modalSpark) { modalSpark.destroy(); modalSpark = null; }
  }
  function openTeamModal(name) {
    var modal = document.getElementById("team-modal");
    if (!modal || !MODAL_DATA) return;
    var t = MODAL_TEAMS[name];
    if (!t) return;
    var c = document.getElementById("team-modal-content");
    var advs = [
      ["Win group", t.win_group], ["Reach R32", t.reach_r32],
      ["Reach R16", t.reach_r16], ["Reach QF", t.reach_qf],
      ["Reach SF", t.reach_sf], ["Reach final", t.reach_final],
      ["Champion", t.model_champion],
    ];
    var html = "<h3>" + esc(t.name) + "</h3>" +
      "<p class='muted'>Elo " + t.elo.toFixed(0) + " (#" + t.rank +
      ") &middot; Group " + t.group + " &middot; market champion " +
      pct(t.market_champion) + "</p><p>" + esc(t.blurb) + "</p>";
    html += "<h4>Advancement</h4>";
    advs.forEach(function (a) {
      var w = Math.min(100, a[1] * 100).toFixed(1);
      html += "<div class='advrow'><span class='lbl'>" + a[0] +
        "</span><span class='bar'><div class='model' style='width:" + w +
        "%'></div></span><span class='pct'>" + pct(a[1]) + "</span></div>";
    });
    html += "<h4>Champion-probability history</h4>" +
      "<div class='chartwrap' style='position:relative;height:150px'>" +
      "<canvas id='modal-spark'></canvas></div>";

    var sched = MODAL_DATA.matches.filter(function (m) {
      return m.home === name || m.away === name;
    });
    html += "<h4>Schedule</h4><div class='modal-sched'>";
    sched.forEach(function (m) {
      var dt = m.date || "";
      if (m.played) {
        html += "<div class='fixture'><span class='date'>" + dt +
          " &middot; final</span><span class='teams'>" + esc(m.home) + " <span class='scorebadge'>" +
          m.score_home + "&ndash;" + m.score_away + "</span> " + esc(m.away) + "</span></div>";
      } else if (m.model) {
        html += "<div class='fixture'><span class='date'>" + dt +
          "</span><span class='teams'>" + esc(m.home) + " vs " + esc(m.away) + "</span>" +
          probrow(m.model, m.home, m.away) +
          (m.explanation ? "<details><summary>Why the model thinks this</summary><p>" +
            esc(m.explanation) + "</p></details>" : "") + "</div>";
      }
    });
    html += "</div>";
    c.innerHTML = html;
    modal.classList.add("open");

    if (typeof Chart !== "undefined") {
      var hist = MODAL_DATA.history;
      var series = hist.map(function (h) {
        var v = (h.model_champion || {})[name];
        return v == null ? null : v * 100;
      });
      var spark = document.getElementById("modal-spark");
      if (modalSpark) modalSpark.destroy();
      modalSpark = new Chart(spark, {
        type: "line",
        data: {
          labels: hist.map(function (h) { return h.as_of; }),
          datasets: [{
            data: series, borderColor: MODEL, backgroundColor: MODEL,
            borderWidth: 2, pointRadius: 3, tension: 0.2,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { title: { display: true, text: "champion %" }, beginAtZero: true } },
        },
      });
    }
  }

  function probrow(m, home, away) {
    var cells = [[home, m.p_home], ["Draw", m.p_draw], [away, m.p_away]];
    var maxi = 0;
    if (m.p_draw >= m.p_home && m.p_draw >= m.p_away) maxi = 1;
    else if (m.p_away >= m.p_home && m.p_away >= m.p_draw) maxi = 2;
    return "<div class='probrow'>" + cells.map(function (c, i) {
      return "<span class='" + (i === maxi ? "fav" : "") + "'>" + esc(c[0]) +
        " " + pct(c[1]) + "</span>";
    }).join("") + "</div>";
  }
})();
