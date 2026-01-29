# Alzheimer’s Disease — Project Highlights  

## 🌟 Overview
A fully modular, clinically grounded data science project analyzing how **brain morphology**, **daily functioning**, and **mental‑health‑related severity indices** relate to Alzheimer’s disease diagnosis.

---

## 🔧 Engineering Highlights
- **End‑to‑end data science pipeline**: preprocessing → statistical analysis → modeling → visualization.
- **Clean, production‑oriented architecture** with reusable Python modules (`preprocessing`, `analysis`, `modeling`, `visualization`, `config`).
- **Strict separation of concerns**: notebooks orchestrate; `src/` contains all logic.
- **Deterministic preprocessing**: numeric conversion, missing‑value handling, raw vs processed data separation.
- **Optional PySpark version** included for scalability, without requiring Spark installation for recruiters.

---

## 🔬 Statistical Highlights
- **ANOVA** to test group differences across diagnosis categories.
- **MANOVA** to evaluate multivariate effects of diagnosis on multiple severity indices.
- **Tukey HSD** for post‑hoc comparisons.
- **Correlation analysis** between structural brain volumes and clinical measures.
- **Chi‑square tests** for categorical associations.

---

## 🤖 Modeling Highlights
- **Linear Regression** and **Random Forest Regression** applied to predict severity indices from brain volumes.
- **Structured model outputs**: R², predictions, feature importances.
- **Transparent evaluation**: models show **very low predictive power**, indicating that:
  - brain morphology **alone** is insufficient to predict individual clinical severity.
  - group‑level differences **do not translate** into individual‑level predictability.

This is a **scientific insight**, not a failure.

---

## 🧠 Key Insights from the Data
- **Daily functioning measures** (e.g., bills, taxes, stove use, travel) show significant group differences across diagnosis categories.
- **Several severity indices** (agitation, anxiety, disinhibition, irritability) differ significantly between diagnostic groups.
- **Brain volumes** correlate with some clinical measures, but correlations are modest.
- **Diagnosis** shows a measurable multivariate effect on combined clinical variables (MANOVA).
- **Predictive models fail** to predict severity scores (R² ≈ 0), reinforcing that:
  - structural brain metrics alone do not capture individual symptom severity.

---

## 🚀 What This Project Delivers
- **4 modular notebooks** covering EDA, modeling, correlations, and subgroup analysis.
- **A clean `src/` architecture** with reusable, testable functions.
- **A professional reporting layer** (`executive_summary.md` + README).
- **Optional PySpark notebook** for scalable preprocessing (not required for main analysis).

---

## 🎯 Why This Project Stands Out
- Strong statistical reasoning.
- Clinical domain awareness.
- Honest interpretation of model performance.
- Clear communication of limitations and insights.
- Senior‑level project structure and reproducibility.

---

## 💬 One‑Sentence Pitch
*A clinically grounded, fully modular data science project showing how brain structure, symptoms, and daily functioning relate in Alzheimer’s disease — and why individual prediction requires more than morphology alone.*
