# Multimodal Machine Learning & Resistome Profiling for In-Hospital Mortality in Aneurysmal Subarachnoid Hemorrhage (aSAH)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Standards: TRIPOD & STROBE](https://img.shields.io/badge/Reporting-TRIPOD%20%7C%20STROBE-green.svg)](https://www.tripod-statement.org/)

An end-to-end, leak-free, multimodal computational pipeline integrating **neurological scoring (GCS)**, **cerebrovascular anatomy**, **systemic inflammatory kinetics (PCT, CRP, TLC)**, and **microbiological resistome profiling** to predict in-hospital mortality, secondary nosocomial infection, and delayed neurological deterioration in patients suffering from aneurysmal subarachnoid hemorrhage (aSAH).

---

## 🔒 Data Privacy & Ethics Compliance Statement

> **Notice**: In strict compliance with healthcare privacy regulations (HIPAA, GDPR, and institutional medical research governance), **all raw patient records, hospital medical record numbers, individual patient names, and protected health information (PHI) have been completely excluded from this public repository**. 
>
> This repository contains the complete analytical pipelines, machine learning algorithms, statistical modeling routines, explainable AI (XAI) frameworks, and bedside scoring logic.

---

## 🔬 Key Clinical Discoveries

1. **Anatomical Blueprint of Vulnerability**:
   - Aneurysm location directly dictates downstream infection susceptibility and clinical trajectory (ICA and ICA-PCOM harboring highest mortality compared to MCA).
2. **The "U-Shaped" CRP Curve & Immunoparalysis**:
   - Extreme systemic inflammation ($\text{CRP} > 60\text{--}62\text{ mg/L}$) and early immunoparalysis / immune exhaustion ($\text{CRP} < 5\text{ mg/L}$) both confer significantly elevated mortality risk.
3. **The "PCT Paradox Zone"**:
   - Borderline Procalcitonin levels ($0.05\text{ to }0.50\text{ ng/mL}$) represent a perilous diagnostic gray zone where clinical hesitation in initiating antimicrobial therapy correlates with septic decompensation.
4. **Pathogen Invasiveness Multiplier**:
   - Deep invasive infections (Bloodstream / CSF) amplify in-hospital death odds by $\sim 10\times$ compared to localized catheter colonization.
5. **The Superbug Landscape**:
   - A $68.7\%$ multidrug resistance (MDR) rate among isolated pathogens to carbapenems and 3rd/4th-generation cephalosporins, leaving Colistin as the primary salvage lifeline.

---

## 📐 Bedside ASIM Risk Score (Aneurysm-SAH Infection-Mortality)

The pipeline derives and internally validates a parsimonious, interpretable point-based clinical scoring system for bedside triage:

| Feature Domain | Clinical Finding | ASIM Points |
| :--- | :--- | :---: |
| **Admission Neurological** | Severe GCS (3–8)<br>Moderate GCS (9–12)<br>Mild GCS (13–15) | **4**<br>**2**<br>**0** |
| **Aneurysm Topography** | Internal Carotid Artery (ICA)<br>ICA-PCOM or ACOM<br>Middle Cerebral Artery (MCA)<br>Other / Posterior | **3**<br>**2**<br>**1**<br>**0** |
| **Infection Profile** | Invasive (Bloodstream / CSF)<br>Non-invasive Positive Culture<br>No Infection | **3**<br>**1**<br>**0** |
| **Biomarker Dynamics** | PCT Paradox Zone ($0.05 \le \text{PCT} \le 0.50\text{ ng/mL}$)<br>CRP Extremes ($<5\text{ or } >60\text{ mg/L}$) | **2**<br>**2** |
| **Demographics** | Age $> 60$ years | **1** |

### Stratified Risk Tiers:
- **Low Risk (0–3 pts)**: Predicted mortality $< 5\%$
- **Moderate Risk (4–6 pts)**: Predicted mortality $5\text{--}15\%$
- **High Risk (7–9 pts)**: Predicted mortality $15\text{--}35\%$
- **Very High Risk ($\ge 10$ pts)**: Predicted mortality $> 40\%$

---

## 🛠️ Modular Pipeline Architecture

The source code in [`pipeline/`](pipeline/) is structured into clean, independently executable modules:

```
pipeline/
├── module1_data_harmonization.py          # GCS string parser (E/V/M), outcome & topography harmonization
├── module2_resistome_engine.py             # Antibiogram cross-sheet linker, ARI scoring, invasive infection mapping
├── module3_feature_engineering.py          # Unified feature matrix, non-linear biomarker kinetics, interaction terms
├── module4_statistical_analysis.py         # Publication Table 1 (Mann-Whitney U, Fisher's Exact) & Table 2
├── module5_predictive_models.py            # Nested 5x5 Stratified CV, Firth's LR, ElasticNet, SVM, RF, XGBoost, LightGBM
├── module6_xai_figures.py                  # Explainable AI: SHAP tree & kernel beeswarm, dependence, Decision Curve Analysis
├── module7_nomogram_risk_score.py          # Bedside ASIM score derivation, calibration curves, clinical nomogram
├── module8_multitask_secondary_endpoints.py# Multi-task prediction (nosocomial infection, delta-GCS deterioration, ICU stay)
├── revised_pipeline_leakfree.py            # Peer-review leak-free overhaul (Admission Tier A vs Dynamic ICU Tier B, EPV parsimony)
├── deep_validation_and_benchmarking.py     # 7-level TRIPOD validation (DeLong, Hosmer-Lemeshow, Brier skill score, permutations)
├── remote_gpu_validation.py                # GPU-accelerated 10,000-permutation test & 5,000-resample bootstrap optimism
├── build_plain_language_overview.py        # Interactive HTML research dashboard for clinicians and reviewers
├── build_visual_ecosystem.py               # Publication-grade vector figure generator
└── convert_to_html.py                      # HTML report compiler
```

---

## ⏱️ Two-Tier Clinical Timing Architecture

To prevent temporal and data leakage, the models are strictly delineated by clinical timing:
* **Tier A: Admission Model ($T_0 \le 24\text{h}$)**:
  * Utilizes strictly admission-available parameters (Initial GCS, age, gender, admission CRP/PCT/TLC, angiographic aneurysm location).
  * Adheres strictly to Events Per Variable (EPV) parsimony constraints.
* **Tier B: Dynamic ICU Model ($T_1 = 48\text{--}72\text{h}$)**:
  * Incorporates longitudinal parameters, secondary nosocomial infection development, microbiological cultures, and resistome dynamics.

---

## 💻 Hardware & Environment

* **CPU**: AMD Ryzen 9 7950X (16 cores, 32 threads)
* **RAM**: 64 GB DDR5-6000
* **GPU**: NVIDIA GeForce RTX 4080 SUPER
* **OS**: Ubuntu 24.04 LTS / Linux x86_64

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Saif-SystemsBio-ClinicalRep/sah-aneurysm-multimodal-study.git
cd sah-aneurysm-multimodal-study
```

### 2. Install Dependencies
```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm shap matplotlib seaborn
```

### 3. Run Pipeline Modules
```bash
# Run leak-free revised models
python pipeline/revised_pipeline_leakfree.py

# Run deep validation & statistical significance benchmarking
python pipeline/deep_validation_and_benchmarking.py

# Generate bedside ASIM nomogram
python pipeline/module7_nomogram_risk_score.py
```

---

## 📜 Citation & License

This project is licensed under the MIT License - see the LICENSE file for details.
