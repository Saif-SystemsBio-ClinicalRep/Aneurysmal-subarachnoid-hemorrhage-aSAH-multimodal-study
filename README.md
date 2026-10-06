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

Discoveries

1. **Anatomical Blueprint of Vulnerability**:
   - Aneurysm location directly dictates downstream infection susceptibility and clinical trajectory (ICA and ICA-PCOM harboring highest mortality compared to MCA).
2. **The "U-Shaped" CRP Curve & Immunoparalysis**:
   - Extreme systemic inflammation and early immunoparalysis / immune exhaustion both confer significantly elevated mortality risk.
3. **The "PCT Paradox Zone"**:
   - Borderline Procalcitonin levels  represent a perilous diagnostic gray zone where clinical hesitation in initiating antimicrobial therapy correlates with septic decompensation.
4. **Pathogen Invasiveness Multiplier**:
   - Deep invasive infections (Bloodstream / CSF) amplify in-hospital death odds  compared to localized catheter colonization.
5. **The Superbug Landscape**e.

---

##  Bedside ASIM Risk Score (Aneurysm-SAH Infection-Mortality)

The pipeline derives and internally validates a parsimonious, interpretable point-based clinical scoring system for bedside triage:

---

##  Modular Pipeline Architecture

The source code in [`pipeline/`](pipeline/) is structured into clean, independently executable modules:

```
pipeline/
├── module1_data_harmonization.py         
├── module2_resistome_engine.py          
├── module3_feature_engineering.py          
├── module4_statistical_analysis.py         
├── module5_predictive_models.py            
├── module6_xai_figures.py               
├── module7_nomogram_risk_score.py          
├── module8_multitask_secondary_endpoints.py
├── revised_pipeline_leakfree.py            
├── deep_validation_and_benchmarking.py    
├── remote_gpu_validation.py                # GPU-accelerated 10,000-permutation test & 5,000-resample bootstrap optimism
├── build_plain_language_overview.py      
├── build_visual_ecosystem.py               
└── convert_to_html.py                      
```

---

##  Two-Tier Clinical Timing Architecture

To prevent temporal and data leakage, the models are strictly delineated by clinical timing:
* **Tier A: Admission Model ($T_0 \le 24\text{h}$)**:
  * Utilizes strictly admission-available parameters (Initial GCS, age, gender, admission CRP/PCT/TLC, angiographic aneurysm location).
  * Adheres strictly to Events Per Variable (EPV) parsimony constraints.
* **Tier B: Dynamic ICU Model ($T_1 = 48\text{--}72\text{h}$)**:
  * Incorporates longitudinal parameters, secondary nosocomial infection development, microbiological cultures, and resistome dynamics.

---


---

##  Getting Started

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
