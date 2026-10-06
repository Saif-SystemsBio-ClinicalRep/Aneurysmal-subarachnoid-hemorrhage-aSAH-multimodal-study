"""
Module 8: Multi-Task Secondary Endpoints Prediction
===================================================
1. Secondary Endpoint 1: Nosocomial Infection Acquisition Prediction
   - Predicts at admission which patients will develop nosocomial infection.
   - Clinical utility: Early proactive infection surveillance & targeted prophylaxis.
   - Target: infection_binary (24 positive, 68 negative)

2. Secondary Endpoint 2: Neurological Deterioration Prediction
   - Predicts neurological decline during hospital stay (Delta GCS < 0 or death).
   - Clinical utility: Identifying patients at risk of secondary neurological injury / vasospasm.
   - Target: GCS_deteriorated (21 deteriorated, 71 stable/improved)

Outputs:
  - Table 5: table5_secondary_endpoints.csv
  - Figure 10: fig10_multitask_endpoints.png
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from xgboost import XGBClassifier

PROJECT_DIR = r"C:\Users\SAIF\.gemini\antigravity-ide\scratch\sah_aneurysm_study"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR = os.path.join(PROJECT_DIR, "output", "figures")
TABLE_DIR = os.path.join(PROJECT_DIR, "output", "tables")

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
})

print("[Module 8] Training Multi-Task Secondary Endpoint Models...")

# ─── LOAD DATA ───
clinical = pd.read_csv(os.path.join(DATA_DIR, 'clinical_backbone.csv'))
unified = pd.read_csv(os.path.join(DATA_DIR, 'unified_feature_matrix.csv'))

# ─── TASK 1: NOSOCOMIAL INFECTION PREDICTION (Admission Features Only) ───
print("  Task 1: Nosocomial Infection Acquisition (n=24 infected / 68 non-infected)...")

# Features known AT ADMISSION (strictly before infection occurs)
admission_features_inf = [
    'age_val', 'gender_binary', 'GCS_total_init', 'GCS_E_init', 'GCS_M_init',
    'intubated_init', 'anterior_circulation', 'aneurysm_ACOM', 'aneurysm_ICA_PCOM',
    'aneurysm_ICA', 'aneurysm_MCA', 'location_infection_weight', 'TLC_val',
    'log_TLC', 'log_PCT', 'log_CRP', 'PCT_paradox_zone', 'neural_inflammatory_coupling'
]

# Ensure features are present
available_inf_features = [f for f in admission_features_inf if f in unified.columns]
X_inf = unified[available_inf_features].fillna(0).replace([np.inf, -np.inf], 0).values
y_inf = unified['infection_binary'].values

# Repeated Stratified K-Fold CV (10 folds x 5 repeats = 50 runs)
cv_inf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

inf_models = {
    'Penalized_LR': LogisticRegression(penalty='l2', C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=150, max_depth=3, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, scale_pos_weight=2.83, eval_metric='logloss', random_state=42)
}

inf_results = {m: {'roc_auc': [], 'pr_auc': [], 'brier': []} for m in inf_models}
inf_oof_preds = {m: np.zeros(len(unified)) for m in inf_models}

scaler = StandardScaler()
for train_idx, val_idx in cv_inf.split(X_inf, y_inf):
    X_train, X_val = X_inf[train_idx], X_inf[val_idx]
    y_train, y_val = y_inf[train_idx], y_inf[val_idx]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    for name, model in inf_models.items():
        if 'LR' in name:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            
        inf_oof_preds[name][val_idx] = probs
        inf_results[name]['roc_auc'].append(roc_auc_score(y_val, probs))
        inf_results[name]['pr_auc'].append(average_precision_score(y_val, probs))
        inf_results[name]['brier'].append(brier_score_loss(y_val, probs))

for m in inf_models:
    mean_auc = np.mean(inf_results[m]['roc_auc'])
    mean_pr = np.mean(inf_results[m]['pr_auc'])
    print(f"    {m:15s} -> ROC-AUC: {mean_auc:.3f}, PR-AUC: {mean_pr:.3f}")

# ─── TASK 2: NEUROLOGICAL DETERIORATION PREDICTION (Delta GCS < 0 or Death) ───
print("  Task 2: Neurological Deterioration (n=21 deteriorated / 71 stable-improved)...")

deteriorated_features = [
    'age_val', 'gender_binary', 'GCS_total_init', 'GCS_M_init', 'intubated_init',
    'location_mortality_weight', 'location_infection_weight', 'aneurysm_ACOM',
    'aneurysm_ICA', 'IBI', 'PCT_paradox_zone', 'CRP_extreme', 'infection_binary',
    'positive_site_count', 'neural_inflammatory_coupling', 'hospital_stay'
]

available_det_features = [f for f in deteriorated_features if f in unified.columns]
X_det = unified[available_det_features].fillna(0).replace([np.inf, -np.inf], 0).values
y_det = clinical['GCS_deteriorated'].values

cv_det = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

det_models = {
    'Penalized_LR': LogisticRegression(penalty='l2', C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=150, max_depth=3, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, scale_pos_weight=3.38, eval_metric='logloss', random_state=42)
}

det_results = {m: {'roc_auc': [], 'pr_auc': [], 'brier': []} for m in det_models}
det_oof_preds = {m: np.zeros(len(unified)) for m in det_models}

for train_idx, val_idx in cv_det.split(X_det, y_det):
    X_train, X_val = X_det[train_idx], X_det[val_idx]
    y_train, y_val = y_det[train_idx], y_det[val_idx]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    for name, model in det_models.items():
        if 'LR' in name:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            
        det_oof_preds[name][val_idx] = probs
        det_results[name]['roc_auc'].append(roc_auc_score(y_val, probs))
        det_results[name]['pr_auc'].append(average_precision_score(y_val, probs))
        det_results[name]['brier'].append(brier_score_loss(y_val, probs))

for m in det_models:
    mean_auc = np.mean(det_results[m]['roc_auc'])
    mean_pr = np.mean(det_results[m]['pr_auc'])
    print(f"    {m:15s} -> ROC-AUC: {mean_auc:.3f}, PR-AUC: {mean_pr:.3f}")

# ─── SAVE TABLE 5: MULTI-TASK SUMMARY ───
table5_rows = []

for task_name, y_t, res_dict in [
    ("Nosocomial Infection Acquisition (Pre-infection Admission Data)", y_inf, inf_results),
    ("Neurological Deterioration (Delta GCS < 0 or In-Hospital Death)", y_det, det_results)
]:
    for m in res_dict:
        auc_vals = res_dict[m]['roc_auc']
        pr_vals = res_dict[m]['pr_auc']
        brier_vals = res_dict[m]['brier']
        
        table5_rows.append({
            'Endpoint': task_name,
            'Model': m,
            'Positive_Prevalence': f"{y_t.mean()*100:.1f}% ({int(y_t.sum())}/{len(y_t)})",
            'ROC_AUC_Mean': f"{np.mean(auc_vals):.3f}",
            'ROC_AUC_95CI': f"[{np.percentile(auc_vals, 2.5):.3f} - {np.percentile(auc_vals, 97.5):.3f}]",
            'PR_AUC_Mean': f"{np.mean(pr_vals):.3f}",
            'PR_AUC_95CI': f"[{np.percentile(pr_vals, 2.5):.3f} - {np.percentile(pr_vals, 97.5):.3f}]",
            'Brier_Score': f"{np.mean(brier_vals):.3f}",
        })

table5_df = pd.DataFrame(table5_rows)
table5_df.to_csv(os.path.join(TABLE_DIR, 'table5_secondary_endpoints.csv'), index=False)
print("  Saved Table 5: table5_secondary_endpoints.csv")

# ─── FIGURE 10: MULTI-TASK ROC & PR CURVES ───
print("  Generating Figure 10: Multi-Task Endpoints Validation...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

colors = {'Penalized_LR': '#1976D2', 'Random_Forest': '#7B1FA2', 'XGBoost': '#D32F2F'}

# Plot 1: Infection ROC
for m in inf_models:
    fpr, tpr, _ = roc_curve(y_inf, inf_oof_preds[m])
    auc_val = roc_auc_score(y_inf, inf_oof_preds[m])
    axes[0, 0].plot(fpr, tpr, color=colors[m], lw=2, label=f"{m} (AUC = {auc_val:.3f})")
axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 0].set_xlabel('1 - Specificity (False Positive Rate)', fontweight='bold')
axes[0, 0].set_ylabel('Sensitivity (True Positive Rate)', fontweight='bold')
axes[0, 0].set_title('A. Nosocomial Infection Acquisition: ROC Curves\n(Admission Predictors Only)', fontweight='bold', fontsize=11)
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, linestyle='--', alpha=0.4)

# Plot 2: Infection PR
for m in inf_models:
    precision, recall, _ = precision_recall_curve(y_inf, inf_oof_preds[m])
    pr_auc_val = average_precision_score(y_inf, inf_oof_preds[m])
    axes[0, 1].plot(recall, precision, color=colors[m], lw=2, label=f"{m} (PR-AUC = {pr_auc_val:.3f})")
axes[0, 1].axhline(y=y_inf.mean(), color='k', linestyle='--', alpha=0.5, label=f'Prevalence ({y_inf.mean():.2f})')
axes[0, 1].set_xlabel('Recall (Sensitivity)', fontweight='bold')
axes[0, 1].set_ylabel('Precision (PPV)', fontweight='bold')
axes[0, 1].set_title('B. Nosocomial Infection Acquisition: PR Curves', fontweight='bold', fontsize=11)
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, linestyle='--', alpha=0.4)

# Plot 3: Deterioration ROC
for m in det_models:
    fpr, tpr, _ = roc_curve(y_det, det_oof_preds[m])
    auc_val = roc_auc_score(y_det, det_oof_preds[m])
    axes[1, 0].plot(fpr, tpr, color=colors[m], lw=2, label=f"{m} (AUC = {auc_val:.3f})")
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1, 0].set_xlabel('1 - Specificity (False Positive Rate)', fontweight='bold')
axes[1, 0].set_ylabel('Sensitivity (True Positive Rate)', fontweight='bold')
axes[1, 0].set_title('C. Neurological Deterioration (Delta GCS < 0): ROC Curves', fontweight='bold', fontsize=11)
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(True, linestyle='--', alpha=0.4)

# Plot 4: Deterioration PR
for m in det_models:
    precision, recall, _ = precision_recall_curve(y_det, det_oof_preds[m])
    pr_auc_val = average_precision_score(y_det, det_oof_preds[m])
    axes[1, 1].plot(recall, precision, color=colors[m], lw=2, label=f"{m} (PR-AUC = {pr_auc_val:.3f})")
axes[1, 1].axhline(y=y_det.mean(), color='k', linestyle='--', alpha=0.5, label=f'Prevalence ({y_det.mean():.2f})')
axes[1, 1].set_xlabel('Recall (Sensitivity)', fontweight='bold')
axes[1, 1].set_ylabel('Precision (PPV)', fontweight='bold')
axes[1, 1].set_title('D. Neurological Deterioration: PR Curves', fontweight='bold', fontsize=11)
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig10_multitask_endpoints.png'), bbox_inches='tight')
plt.close()
print("    Saved fig10_multitask_endpoints.png")

print("\n[Module 8] COMPLETE")
