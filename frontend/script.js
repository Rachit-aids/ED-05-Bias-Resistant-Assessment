const navItems = document.querySelectorAll(".nav-item");
const sections = document.querySelectorAll(".page-section");
const pageTitle = document.getElementById("pageTitle");

const titles = {
  assessment: "Semantic Answer Analyzer",
  robustness: "Counterfactual Stability",
  metrics: "Model Quality Dashboard",
  about: "About the ED-05 System"
};

function showSection(id) {
  sections.forEach(s => s.classList.toggle("active-section", s.id === id));
  navItems.forEach(n => n.classList.toggle("active", n.dataset.section === id));
  pageTitle.textContent = titles[id] || "ED-05";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

navItems.forEach(item => item.addEventListener("click", () => showSection(item.dataset.section)));
document.querySelectorAll("[data-jump]").forEach(btn => btn.addEventListener("click", () => showSection(btn.dataset.jump)));

document.getElementById("themeBtn").addEventListener("click", () => {
  document.body.classList.toggle("light");
});

const demoPredictions = [
  { label:"CORRECT", title:"Correct", confidence:94, p:[94.2,2.8,3.0], cls:"correct",
    text:"The response captures the core causal relationship and does not introduce a contradiction." },
  { label:"CONTRADICTORY", title:"Contradictory", confidence:81, p:[8.2,81.0,10.8], cls:"contradictory",
    text:"The response addresses the topic but states a relationship that conflicts with the reference explanation." },
  { label:"INCORRECT", title:"Incorrect", confidence:88, p:[4.0,8.0,88.0], cls:"incorrect",
    text:"The answer does not demonstrate the requested concept and lacks the necessary semantic relationship." }
];

function setPrediction(result) {
  const { label, title, confidence, p, cls, text } = result;
  const badge = document.getElementById("resultBadge");
  badge.textContent = label;
  badge.className = "result-badge " + cls;

  document.getElementById("predictionTitle").textContent = title;
  document.getElementById("predictionText").textContent = text;
  document.getElementById("confidence").textContent = confidence;

  const fields = [
    ["pCorrect", "barCorrect", p[0]],
    ["pContradictory", "barContradictory", p[1]],
    ["pIncorrect", "barIncorrect", p[2]]
  ];
  fields.forEach(([t,b,v]) => {
    document.getElementById(t).textContent = v.toFixed(1) + "%";
    document.getElementById(b).style.width = v + "%";
  });

  document.getElementById("scoreRing").style.background =
    `conic-gradient(var(--accent) ${confidence * 3.6}deg, #2c3040 0deg)`;
}

document.getElementById("analyzeBtn").addEventListener("click", () => {
  const answer = document.getElementById("answer").value.trim().toLowerCase();
  let result = demoPredictions[0];

  if (!answer) {
    result = demoPredictions[2];
  } else if (/(never|opposite|decreases|false|wrong)/.test(answer)) {
    result = demoPredictions[1];
  } else if (answer.length < 35) {
    result = demoPredictions[2];
  } else {
    result = demoPredictions[0];
  }

  setPrediction(result);
});

document.getElementById("clearBtn").addEventListener("click", () => {
  document.getElementById("answer").value = "";
  document.getElementById("question").value = "";
  document.getElementById("reference").value = "";
  setPrediction(demoPredictions[0]);
});

// Default state
setPrediction(demoPredictions[0]);
