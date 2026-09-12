(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

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
    sections.forEach((section) => {
      section.classList.toggle("active-section", section.id === id);
    });
    navItems.forEach((item) => {
      item.classList.toggle("active", item.dataset.section === id);
    });
    if (pageTitle) pageTitle.textContent = titles[id] || "ED-05";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  navItems.forEach((item) => {
    item.addEventListener("click", () => showSection(item.dataset.section));
  });

  document.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => showSection(btn.dataset.jump));
  });

  $("themeBtn")?.addEventListener("click", () => {
    document.body.classList.toggle("light");
  });

  const demoPredictions = {
    correct: {
      label: "CORRECT",
      title: "Correct",
      confidence: 94,
      p: [94.2, 2.8, 3.0],
      cls: "correct",
      text: "The response captures the core semantic relationship and is consistent with the supplied reference answer."
    },
    contradictory: {
      label: "CONTRADICTORY",
      title: "Contradictory",
      confidence: 84,
      p: [7.0, 84.0, 9.0],
      cls: "contradictory",
      text: "The response addresses the topic but contains a claim that conflicts with the expected relationship."
    },
    incorrect: {
      label: "INCORRECT",
      title: "Incorrect",
      confidence: 91,
      p: [3.0, 6.0, 91.0],
      cls: "incorrect",
      text: "The response does not provide enough of the required semantic content to support the expected answer."
    }
  };

  // Browser-only demo classifier. It is deliberately simple and is NOT the final ML model.
  // It uses contradiction cues first, then overlap with the reference answer, so unrelated
  // or clearly wrong demo answers no longer default to CORRECT.
  const STOP_WORDS = new Set([
    "the","a","an","and","or","but","to","of","in","on","for","with","is","are","was","were",
    "be","being","been","that","this","it","its","as","by","from","at","than","so","because",
    "into","about","their","there","they","them","we","you","your","i","my","me","do","does",
    "did","has","have","had","will","would","can","could","should","may","might","more","most",
    "very","generally","what","why","how","which","who","when","where","not"
  ]);

  const CONTRADICTION_PATTERNS = [
    /\bdecreases?\b/i,
    /\bslower\b/i,
    /\bno effect\b/i,
    /\bhas no effect\b/i,
    /\bdoes not increase\b/i,
    /\bdoesn't increase\b/i,
    /\bopposite\b/i,
    /\bnever\b/i,
    /\bfalse\b/i,
    /\bincorrect\b/i,
    /\bwrong\b/i,
    /\bnot true\b/i,
    /\bthe reaction becomes slower\b/i
  ];

  const INCORRECT_PATTERNS = [
    /\bi don't know\b/i,
    /\bi do not know\b/i,
    /\bnot related\b/i,
    /\bno relation\b/i,
    /\birrelevant\b/i,
    /\brandom\b/i
  ];

  function normalize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function contentTokens(text) {
    return new Set(
      normalize(text)
        .split(" ")
        .filter(Boolean)
        .filter((token) => token.length > 2 && !STOP_WORDS.has(token))
    );
  }

  function classifyDemo(answer, reference) {
    const cleanAnswer = normalize(answer);
    if (!cleanAnswer) return demoPredictions.incorrect;

    if (CONTRADICTION_PATTERNS.some((pattern) => pattern.test(cleanAnswer))) {
      return demoPredictions.contradictory;
    }

    if (INCORRECT_PATTERNS.some((pattern) => pattern.test(cleanAnswer))) {
      return demoPredictions.incorrect;
    }

    const answerTokens = contentTokens(answer);
    const referenceTokens = contentTokens(reference);

    if (answerTokens.size === 0) {
      return demoPredictions.incorrect;
    }

    if (referenceTokens.size > 0) {
      let overlap = 0;
      answerTokens.forEach((token) => {
        if (referenceTokens.has(token)) overlap += 1;
      });

      const overlapRatio = overlap / Math.max(1, answerTokens.size);

      // Very low content overlap => unrelated/incorrect demo answer.
      if (overlapRatio < 0.12 && answerTokens.size >= 4) {
        return demoPredictions.incorrect;
      }

      // Strong content overlap => correct demo answer.
      if (overlapRatio >= 0.24) {
        return demoPredictions.correct;
      }
    }

    // Short responses are treated as uncertain/incorrect for the demo.
    if (cleanAnswer.length < 25) {
      return demoPredictions.incorrect;
    }

    // Neutral fallback: do not artificially inflate CORRECT.
    return {
      ...demoPredictions.incorrect,
      confidence: 72,
      p: [12.0, 16.0, 72.0],
      text: "The browser demo could not establish enough semantic overlap with the reference answer."
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

    const fields = [
      ["pCorrect", "barCorrect", p[0]],
      ["pContradictory", "barContradictory", p[1]],
      ["pIncorrect", "barIncorrect", p[2]]
    ];

    fields.forEach(([textId, barId, value]) => {
      if ($(textId)) $(textId).textContent = `${value.toFixed(1)}%`;
      if ($(barId)) $(barId).style.width = `${value}%`;
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

    // Small delay makes the interaction visibly responsive in the demo.
    window.setTimeout(() => {
      const result = classifyDemo(answer, reference);
      setPrediction(result);

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
    setPrediction(demoPredictions.incorrect);
  });

  setPrediction(demoPredictions.correct);
})();
