# ED-05 Frontend

Professional frontend prototype for the ED-05 Bias-Resistant Short-Answer Assessment project.

## Files

- `index.html` — dashboard UI
- `style.css` — responsive styling
- `script.js` — navigation and demo interactions

## Run locally

Open `index.html` directly in a browser, or serve the folder with:

```bash
python -m http.server 5500
```

Then open `http://localhost:5500`.

## Backend integration

The current UI uses deterministic demo predictions so it can be previewed without the Python backend.

For the final integrated version, replace the demo logic in `script.js` with a request such as:

```javascript
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question,
    reference_answer,
    student_answer
  })
});

const result = await response.json();
```

Expected result fields can include:

```json
{
  "predicted_class": "correct",
  "probabilities": {
    "correct": 0.942,
    "contradictory": 0.028,
    "incorrect": 0.030
  },
  "confidence": 0.942,
  "evidence": [],
  "counterfactuals": []
}
```
