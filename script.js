(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // Navigation
  const navItems = document.querySelectorAll(".nav-item");
  const sections = document.querySelectorAll(".page-section");
  const pageTitle = $("pageTitle");

  const titles = {
    assessment: "Semantic Answer Analyzer",
    robustness: "Counterfactual Stability",
    metrics: "Model Quality Dashboard",
    about: "About the ED-05 System"
  };

  function showSection(id) {
    sections.forEach(s => s.classList.toggle("active-section", s.id === id));
    navItems.forEach(n => n.classList.toggle("active", n.dataset.section === id));
    if (pageTitle) pageTitle.textContent = titles[id] || "ED-05";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  navItems.forEach(item => item.addEventListener("click", () => showSection(item.dataset.section)));
  document.querySelectorAll("[data-jump]").forEach(btn => {
    btn.addEventListener("click", () => showSection(btn.dataset.jump));
  });

  $("themeBtn")?.addEventListener("click", () => document.body.classList.toggle("light"));

  const base = {
    correct: {
      label: "CORRECT", title: "Correct", confidence: 94,
      p: [94.2, 2.8, 3.0], cls: "correct",
      text: "The response captures the key semantic relationship and is consistent with the reference answer."
    },
    contradictory: {
      label: "CONTRADICTORY", title: "Contradictory", confidence: 87,
      p: [5.0, 87.0, 8.0], cls: "contradictory",
      text: "The response contains a statement that conflicts with the expected relationship in the reference answer."
    },
    incorrect: {
      label: "INCORRECT", title: "Incorrect", confidence: 91,
      p: [3.0, 6.0, 91.0], cls: "incorrect",
      text: "The response does not contain enough of the required semantic content to support the reference answer."
    }
  };

  const STOP = new Set([
    "the","a","an","and","or","but","to","of","in","on","for","with","is","are","was","were",
    "be","been","being","that","this","it","its","as","by","from","at","than","so","because",
    "into","about","their","there","they","them","we","you","your","i","my","me","do","does",
    "did","has","have","had","will","would","can","could","should","may","might","more","most",
    "very","generally","what","why","how","which","who","when","where","not","during","mainly",
    "means","use","uses","using","happens","happen","becomes","become"
  ]);

  const CONTRADICTION_PATTERNS = [
    /\bdo not need\b/i, /\bdoes not need\b/i, /\bdon't need\b/i,
    /\bno effect\b/i, /\bhas no effect\b/i,
    /\bdecrease(?:s|d)?\b/i, /\bdecreases?\b/i, /\bslower\b/i,
    /\bslows?\b/i, /\breduces?\b/i, /\breduction\b/i,
    /\bopposite\b/i, /\bnever\b/i, /\bfalse\b/i, /\bwrong\b/i,
    /\bnot required\b/i, /\bis unnecessary\b/i, /\bisn't necessary\b/i,
    /\bis not necessary\b/i, /\bwithout .* need\b/i
  ];

  const INCORRECT_CUES = [
    /\bdo not know\b/i, /\bi don't know\b/i,
    /\bno relation\b/i, /\bnot related\b/i, /\birrelevant\b/i,
    /\brandom\b/i
  ];

  function normalize(text) {
    return text.toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function tokens(text) {
    return normalize(text)
      .split(" ")
      .filter(Boolean)
      .filter(t => t.length > 2 && !STOP.has(t));
  }

  function unique(tokensArr) {
    return [...new Set(tokensArr)];
  }

  // A small browser demo only. This is deliberately not represented as the
  // final ML model: the production UI will call the Python/FastAPI endpoint.
  function classifyDemo(answer, reference) {
    const a = normalize(answer);
    const r = normalize(reference);

    if (!a) return base.incorrect;

    // Explicit contradiction gets priority over lexical similarity.
    if (CONTRADICTION_PATTERNS.some(rx => rx.test(a))) return base.contradictory;
    if (INCORRECT_CUES.some(rx => rx.test(a))) return base.incorrect;

    const aTokens = unique(tokens(answer));
    const rTokens = unique(tokens(reference));

    if (aTokens.length < 2) return base.incorrect;

    const rSet = new Set(rTokens);
    const overlap = aTokens.filter(t => rSet.has(t));
    const overlapRatio = overlap.length / Math.max(1, aTokens.length);

    // Protect against answers that repeat one topical keyword but omit the core meaning.
    // A generic answer needs either several independent reference concepts or strong coverage.
    if (overlap.length >= 4 || (overlap.length >= 3 && overlapRatio >= 0.45)) {
      return base.correct;
    }

    // For short paraphrases, two meaningful matches can still be enough when they cover
    // a substantial portion of the response.
    if (overlap.length >= 2 && overlapRatio >= 0.55) {
      return base.correct;
    }

    return {
      ...base.incorrect,
      confidence: 89,
      p: [4.0, 7.0, 89.0],
      text: overlap.length
        ? "The response shares some topic vocabulary, but it does not establish enough of the reference answer's core semantic content."
        : "The response has little or no semantic overlap with the supplied reference answer."
    };
  }

  function setPrediction(result) {
    const { label, title, confidence, p, cls, text } = result;

    const badge = $("resultBadge");
    if (badge) {
      badge.textContent = label;
      badge.className = `result-badge ${cls}`;
    }

    if ($("predictionTitle")) $("predictionTitle").textContent = title;
    if ($("predictionText")) $("predictionText").textContent = text;
    if ($("confidence")) $("confidence").textContent = Math.round(confidence);

    [
      ["pCorrect", "barCorrect", p[0]],
      ["pContradictory", "barContradictory", p[1]],
      ["pIncorrect", "barIncorrect", p[2]]
    ].forEach(([t, b, v]) => {
      if ($(t)) $(t).textContent = `${Number(v).toFixed(1)}%`;
      if ($(b)) $(b).style.width = `${v}%`;
    });

    if ($("scoreRing")) {
      $("scoreRing").style.background =
        `conic-gradient(var(--accent) ${confidence * 3.6}deg, #2c3040 0deg)`;
    }
  }

  $("analyzeBtn")?.addEventListener("click", (event) => {
    event.preventDefault();

    const button = $("analyzeBtn");
    const answer = $("answer")?.value?.trim() || "";
    const reference = $("reference")?.value?.trim() || "";

    if (button) {
      button.disabled = true;
      button.innerHTML = `Analyzing <span>…</span>`;
    }

    setTimeout(() => {
      setPrediction(classifyDemo(answer, reference));

      if (button) {
        button.disabled = false;
        button.innerHTML = `Analyze answer <span>→</span>`;
      }
    }, 250);
  });

  $("clearBtn")?.addEventListener("click", () => {
    if ($("answer")) $("answer").value = "";
    if ($("question")) $("question").value = "";
    if ($("reference")) $("reference").value = "";
    setPrediction(base.incorrect);
  });

  // Initial state
  setPrediction(base.correct);
})();
