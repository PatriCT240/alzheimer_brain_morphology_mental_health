<div style="
    width: 100%;
    padding: 30px 20px;
    background: linear-gradient(90deg, #004e92, #000428);
    border-radius: 8px;
    text-align: center;
    color: white;
    font-family: Arial, sans-serif;
">
    <h1 style="margin: 0; font-size: 32px; font-weight: 700;">
        Exploring Brain Morphology & Mental Health in Alzheimer’s Disease
    </h1>
    <p style="margin: 10px 0 0; font-size: 18px; font-weight: 300;">
        A multimodal clinical and brain morphology analysis
    </p>
</div>

**Repository:** `alzheimer_brain_morphology_mental_health`  
**Goal:** Identify statistical and predictive relationships between brain morphology, daily functioning, mental health severity, and Alzheimer’s diagnosis.

---

## 📌 Executive Summary

This project investigates how structural brain measures and mental‑health‑related clinical indices relate to Alzheimer’s disease severity and diagnosis. It integrates:

- clinical severity indices  
- daily activity performance  
- brain volume measurements  
- group‑based statistical tests (ANOVA, MANOVA, Tukey)  
- predictive modeling (Linear Regression, Random Forest)  
- correlation analysis between neuroimaging and clinical variables  

The entire codebase follows a **clean, modular, industry‑grade architecture**, with strict separation between computation, visualization, and result storage.

---

## 🗂️ Project Structure
alzheimer_brain_morphology_mental_health/
│
├── data/
│   ├── raw/                     # Raw data
│   └── processed/               # Cleaned and transformed data
│
├── notebooks/                
│   ├── 01_EDA_relationships.ipynb         # Exploratory Data Analysis
│   ├── 02_modeling.ipynb                  # Predictive modeling
│   ├── 03_correlation_analysis.ipynb      # Statistical correlations
│   └── 04_subgroups.ipynb                 # Subgroup-based analysis
│
├── src/
│   ├── analysis.py            # ANOVA, MANOVA, correlations, chi-square, Tukey
│   ├── config.py              # Global variables, column groups, parameters
│   ├── modeling.py            # LR, RF, model comparison
│   ├── preprocessing.py       # Data loading, cleaning, missing values
│   └── visualization.py       # Modular plotting utilities
│
├── reports/
│   ├── figures/               # Generated plots
│   ├── tables/                # Statistical outputs (ANOVA, Tukey, MANOVA…)
│   └── executive_summary.md   # High-level summary of findings
│
├── spark/                    # Optional PySpark version (non‑required)
│   ├── spark_version_optional.ipynb
│   └── environment.yml
│
├── requirements.txt
├── README.md
├── Project_Highlights.md
└── .gitignore


---

## ⚡ About the PySpark Version (Optional)

The `spark/` folder contains:

- `spark_version_optional.ipynb`  
- `environment.yml`  

This reflects the **original PySpark implementation** of the project.

However, the repository has been intentionally adapted so that:

- **recruiters do NOT need to install PySpark**  
- **the main workflow runs entirely in pandas**  
- **Spark is provided only as an optional, advanced version**  

This design ensures:

- lightweight execution  
- compatibility with standard Python environments  
- demonstration of scalability to distributed systems without imposing Spark as a dependency  

The Spark notebook showcases the ability to scale the analysis to large datasets while keeping the main workflow accessible.

---

## 🔬 Analytical Workflow

### 1. Data Preprocessing
- CSV loading  
- numeric conversion  
- missing value handling  
- clean separation between raw and processed data  

### 2. Exploratory Data Analysis
- detection of severity and activity columns  
- descriptive statistics  
- visualization of group differences  

### 3. Statistical Analysis
- Combined subgroup ANOVA
- Tukey HSD post‑hoc comparisons  
- MANOVA for multivariate effects  
- Correlation matrices and p‑value heatmaps
- ANOVA with covariates (`female`, `educ`)  

### 4. Predictive Modeling
- Linear Regression  
- Random Forest  
- R² comparison across severity indices  
- Feature importance extraction  

### 5. Visualization
- Regression plots  
- Barplots and group‑mean plots  
- Feature importance charts
- Heatmaps (correlations, p‑values)  

All visualizations are optional to save (`save_path=None` by default), ensuring full modularity.

---

## 🧩 Code Architecture

### `preprocessing.py`
- data loading  
- numeric conversion  
- missing value handling  
- summary statistics  

### `analysis.py`
- Pearson correlations
- ANOVA, MANOVA  
- Chi‑square tests  
- Tukey post‑hoc  
- Returns clean DataFrames  

### `modeling.py`
- Linear Regression  
- Random Forest  
- Model comparison utilities  
- Returns structured dictionaries (model, R², predictions, importances)

### `visualization.py`
- Modular plotting utilities  
- Saving only when `save_path` is provided  
- Consistent, publication‑ready aesthetics

### `config.py`
- Column groups  
- Brain volume categories  
- Statistical parameters  
- Default covariates  

---

## ⚙️ Requirements

Python 3.10+
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy

PySpark is optional and only required for the notebook inside `/spark/`.

---

## ▶️ How to Run

1. Clone the repository  
2. Install dependencies  
3. Run notebooks in order (`01_...` → `02_...` → `03_...` → `04_...`)  
4. Results will be saved automatically to `reports/tables/` and `reports/figures/`

---

## 📈 Key Findings

- Several daily activities show significant differences by gender and education.  
- Women exhibit stronger impairments in **stove use**, **attention**, **social events**, and **games**.  
- Strong correlations exist between temporal/hippocampal volumes and clinical severity.  
- Random Forest consistently outperforms Linear Regression in predictive accuracy.  

---

## 📄 License

MIT License.

## 👩‍⚕️ Author
Patricia C. Torrell  
Clinical Data Analyst transitioning into Data Analytics
Focused on clinical modeling, reproducible pipelines, and interpretable ML.

**LinkedIn**: https://www.linkedin.com/in/patricia-c-torrell
**GitHub**: https://github.com/PatriCT240.github.io

---

## 🔑 Key Takeaways for Recruiters

- **Industry‑grade project architecture** with strict modular separation (`preprocessing`, `analysis`, `modeling`, `visualization`, `config`).
- **Reproducible and transparent workflow**, with all saving logic handled from notebooks and no side‑effects inside `src/`.
- **Advanced statistical expertise**: ANOVA, MANOVA, Tukey HSD, correlation matrices, chi‑square tests, subgroup analysis.
- **Predictive modeling proficiency** using Linear Regression and Random Forest, with structured model outputs and feature importance analysis.
- **Clinical domain understanding**, working with severity indices, daily functioning measures, and structural brain morphology metrics.
- **Clean data engineering practices**: raw vs processed data separation, numeric conversion, missing‑value strategies, and standardized preprocessing.
- **Professional visualization layer** with modular, publication‑ready plots and optional saving paths.
- **PySpark‑ready pipeline** included as an optional scalable version, demonstrating ability to work with distributed systems without imposing heavy dependencies.
- **Clear communication and documentation**, including an executive summary, project highlights, and a recruiter‑friendly README.






