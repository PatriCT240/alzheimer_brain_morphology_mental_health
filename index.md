# Alzheimer’s Disease: Brain Morphology, Daily Functioning, and Clinical Severity  
*A clinically grounded data science project*

This project investigates how **brain structural metrics**, **daily functional abilities**, and **clinical severity indices** vary across Alzheimer’s disease diagnostic groups.  
It combines rigorous statistical analysis with a clean, modular engineering architecture suitable for both scientific and recruiter-facing audiences.

---

## 🔍 Project Overview

Alzheimer’s disease is a complex neurodegenerative condition where cognitive decline, functional impairment, and brain atrophy evolve at different rates.  
This project explores three core questions:

1. **Do brain volumes, daily activities, and symptom severity differ across diagnostic groups?**  
2. **Are brain morphology metrics correlated with clinical severity?**  
3. **Can structural brain measures predict individual symptom severity?**

The analysis is grounded in clinical reasoning and supported by a fully reproducible data science pipeline.

---

## 🧠 Key Findings

### **1. Strong group-level differences**  
ANOVA and MANOVA analyses reveal clear differences across diagnostic groups in:
- Daily functioning  
- Symptom severity  
- Several brain volume measures  

### **2. Weak correlations between brain morphology and severity**  
Correlations between structural volumes and clinical severity are modest, suggesting that morphology alone does not explain symptom burden.

### **3. Very low predictive power**  
Baseline models (Linear Regression, Random Forest) show:
- **Low R² values**
- **Poor generalization**
- **High error variance**

➡️ **Brain morphology alone cannot predict individual clinical severity.**

This aligns with clinical evidence: Alzheimer’s symptoms emerge from a combination of structural, functional, and neurobiological factors.

---

## 🧪 Methods & Pipeline

The project follows a fully modular, reproducible workflow:

### **1. Preprocessing**
- Deterministic cleaning  
- Numeric conversion  
- Missing value handling  
- Export of clean datasets  

### **2. Statistical Analysis**
- Pearson correlations  
- ANOVA & MANOVA  
- Tukey HSD post‑hoc tests  
- Chi‑square tests for categorical variables  

### **3. Modeling**
- Linear Regression  
- Random Forest Regressor  
- Model comparison and evaluation  

### **4. Visualization**
- Group mean plots  
- Side‑by‑side bar charts  
- Correlation heatmaps  
- Diagnostic group comparisons  

All figures and tables are generated automatically and stored in:
reports/figures/
reports/tables/


---

## 📁 Repository Structure
alzheimer_brain_morphology_mental_health/
│
├── notebooks/        # Clean, modular Jupyter notebooks
├── src/              # Reusable analysis modules
├── data/             # Raw and processed datasets
├── reports/
│   ├── figures/      # Generated plots
│   └── tables/       # Generated statistical outputs
└── README.md          # Technical project description


---

## 📊 Figures & Tables

All visual outputs and statistical tables are available directly in the repository:

- **reports/figures** → diagnostic plots, group comparisons, volume charts  
- **reports/tables** → ANOVA results, correlations, Tukey tests, model metrics  

These are generated automatically when running the notebooks.

---

## 🎯 Purpose of the Project

This project is designed to:

- Demonstrate **clinical data science expertise**  
- Showcase **statistical rigor** and **interpretability**  
- Provide a **clean engineering architecture** aligned with industry standards  
- Communicate insights clearly to both technical and clinical audiences  

It is part of a broader portfolio focused on **clinical analytics**, **brain health**, and **evidence‑based modeling**.

---

## 👩‍💻 Author

**Patri**  
Clinical Data Analyst & Data Scientist  
Focused on clinical modeling, and reproducible scientific workflows.

---

## 📬 Contact

For collaboration or inquiries, feel free to reach out via GitHub.


