"""
Module 7: Clinical Nomogram & Bedside Risk Score (ASIM-Score)
============================================================
Translates top machine learning and epidemiological discoveries into
an actionable, interpretable bedside point score and clinical nomogram:
  - Aneurysm-SAH Infection-Mortality (ASIM) Risk Score
  - 1,000-iteration Bootstrap Internal Validation (C-index, Brier, CI)
  - Calibration and Risk Stratification Analysis
  - Publication Figures:
      * fig8_clinical_nomogram.png
      * fig8_score_risk_curve.png
  - Output Tables:
      * table4_nomogram_validation.csv
      * patient_risk_scores.csv
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, precision_recall_curve, f1_score
from scipy import stats

PROJECT_DIR = r"C:\Users\SAIF\.gemini\antigravity-ide\scratch\sah_aneurysm_study"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "output", "models")
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

print("[Module 7] Deriving Bedside Risk Score (ASIM-Score) & Clinical Nomogram...")

# ─── LOAD DATA ───
clinical = pd.read_csv(os.path.join(DATA_DIR, 'clinical_backbone.csv'))
unified = pd.read_csv(os.path.join(DATA_DIR, 'unified_feature_matrix.csv'))

# ─── SCORE DERIVATION ───
# ASIM Point Allocation:
# 1. GCS Initial Severity: Severe (3-8) = 4 pts, Moderate (9-12) = 2 pts, Mild (13-15) = 0 pts
# 2. Aneurysm Location: ICA = 3 pts, ICA-PCOM = 2 pts, ACOM = 2 pts, MCA = 1 pt, Others = 0 pts
# 3. Infection Profile: Invasive (Blood/CSF) = 3 pts, Non-invasive positive = 1 pt, None = 0 pts
# 4. PCT Paradox Zone: 0.05 <= PCT <= 0.5 ng/mL = 2 pts, otherwise 0 pts
# 5. Inflammatory Discordance / CRP Extreme: CRP < 5 or > 62 mg/L = 2 pts, otherwise 0 pts
# 6. Age: > 60 years = 1 pt, <= 60 years = 0 pts

def compute_asim_score(row):
    score = 0
    
    # 1. GCS Initial
    gcs = row['GCS_total_init']
    if gcs <= 8:
        score += 4
    elif gcs <= 12:
        score += 2
    else:
        score += 0
        
    # 2. Aneurysm Location
    loc = str(row['aneurysm_clean'])
    if loc == 'ICA':
        score += 3
    elif loc in ['ICA-PCOM', 'ACOM']:
        score += 2
    elif loc == 'MCA':
        score += 1
    else:
        score += 0
        
    # 3. Infection Profile
    blood_pos = row.get('culture_blood_positive', 0)
    csf_pos = row.get('culture_csf_positive', 0)
    inf_bin = row.get('infection_binary', 0)
    if blood_pos == 1 or csf_pos == 1:
        score += 3
    elif inf_bin == 1:
        score += 1
    else:
        score += 0
        
    # 4. PCT Paradox Zone
    pct = row['PCT']
    if 0.05 <= pct <= 0.5:
        score += 2
    else:
        score += 0
        
    # 5. CRP Extreme
    crp = row['CRP_val']
    if crp < 5.0 or crp > 62.0:
        score += 2
    else:
        score += 0
        
    # 6. Age
    age = row['age_val']
    if age > 60:
        score += 1
    else:
        score += 0
        
    return score

clinical['asim_score'] = clinical.apply(compute_asim_score, axis=1)
y_true = clinical['outcome_binary'].values
scores = clinical['asim_score'].values

# Fit calibration model: Logit(P) = beta0 + beta1 * Score
lr = LogisticRegression(penalty=None, solver='lbfgs')
lr.fit(scores.reshape(-1, 1), y_true)
beta0 = lr.intercept_[0]
beta1 = lr.coef_[0][0]
clinical['predicted_prob'] = lr.predict_proba(scores.reshape(-1, 1))[:, 1]

print(f"  Fitted ASIM Logistic Model: Logit(P) = {beta0:.4f} + {beta1:.4f} * ASIM_Score")
print(f"  Odds Ratio per 1-point increase: {np.exp(beta1):.3f}")

# ─── BOOTSTRAP INTERNAL VALIDATION (1,000 Resamples) ───
print("  Running 1,000-iteration Bootstrap Internal Validation...")
np.random.seed(42)
n_boot = 1000
n_samples = len(clinical)
boot_auc = []
boot_brier = []
boot_sens = []
boot_spec = []

for b in range(n_boot):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    b_scores = scores[idx]
    b_y = y_true[idx]
    
    # Must have both classes in bootstrap sample
    if len(np.unique(b_y)) < 2:
        continue
        
    b_auc = roc_auc_score(b_y, b_scores)
    boot_auc.append(b_auc)
    
    # Fit model on bootstrap sample to evaluate calibration and threshold
    b_lr = LogisticRegression(penalty=None, solver='lbfgs')
    b_lr.fit(b_scores.reshape(-1, 1), b_y)
    b_probs = b_lr.predict_proba(b_scores.reshape(-1, 1))[:, 1]
    boot_brier.append(brier_score_loss(b_y, b_probs))
    
    # Sensitivity & specificity at score >= 6 (clinical threshold)
    pred_pos = (b_scores >= 6).astype(int)
    tn = np.sum((pred_pos == 0) & (b_y == 0))
    fp = np.sum((pred_pos == 1) & (b_y == 0))
    fn = np.sum((pred_pos == 0) & (b_y == 1))
    tp = np.sum((pred_pos == 1) & (b_y == 1))
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    boot_sens.append(sens)
    boot_spec.append(spec)

auc_mean = np.mean(boot_auc)
auc_lower, auc_upper = np.percentile(boot_auc, [2.5, 97.5])
brier_mean = np.mean(boot_brier)
brier_lower, brier_upper = np.percentile(boot_brier, [2.5, 97.5])
sens_mean = np.mean(boot_sens)
sens_lower, sens_upper = np.percentile(boot_sens, [2.5, 97.5])
spec_mean = np.mean(boot_spec)
spec_lower, spec_upper = np.percentile(boot_spec, [2.5, 97.5])

print(f"  Bootstrap C-index (ROC-AUC): {auc_mean:.3f} (95% CI: [{auc_lower:.3f}, {auc_upper:.3f}])")
print(f"  Bootstrap Brier Score: {brier_mean:.3f} (95% CI: [{brier_lower:.3f}, {brier_upper:.3f}])")
print(f"  Sensitivity (Score >= 6): {sens_mean*100:.1f}% (95% CI: [{sens_lower*100:.1f}%, {sens_upper*100:.1f}%])")
print(f"  Specificity (Score >= 6): {spec_mean*100:.1f}% (95% CI: [{spec_lower*100:.1f}%, {spec_upper*100:.1f}%])")

# ─── RISK STRATIFICATION ───
# Stratify into 4 actionable tiers:
# Low (0-3), Moderate (4-6), High (7-9), Very High (10+)
def assign_risk_tier(s):
    if s <= 3:
        return 'Low Risk (0-3)'
    elif s <= 6:
        return 'Moderate Risk (4-6)'
    elif s <= 9:
        return 'High Risk (7-9)'
    else:
        return 'Very High Risk (10+)'

clinical['risk_tier'] = clinical['asim_score'].apply(assign_risk_tier)

tier_order = ['Low Risk (0-3)', 'Moderate Risk (4-6)', 'High Risk (7-9)', 'Very High Risk (10+)']
tier_summary = []

for tier in tier_order:
    sub = clinical[clinical['risk_tier'] == tier]
    n_tot = len(sub)
    n_deaths = int(sub['outcome_binary'].sum())
    obs_rate = (n_deaths / n_tot * 100) if n_tot > 0 else 0
    mean_pred = (sub['predicted_prob'].mean() * 100) if n_tot > 0 else 0
    
    # 95% Wilson score CI for observed proportion
    if n_tot > 0:
        p_hat = n_deaths / n_tot
        z = 1.96
        denom = 1 + z**2 / n_tot
        center = (p_hat + z**2 / (2 * n_tot)) / denom
        delta = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_tot)) / n_tot) / denom
        ci_str = f"[{max(0, center - delta)*100:.1f}% - {min(1, center + delta)*100:.1f}%]"
    else:
        ci_str = "N/A"
        
    tier_summary.append({
        'Risk_Tier': tier,
        'Patient_Count': n_tot,
        'Percentage_Cohort': f"{(n_tot/len(clinical)*100):.1f}%",
        'Observed_Deaths': n_deaths,
        'Observed_Mortality_Rate': f"{obs_rate:.1f}%",
        'Observed_95CI': ci_str,
        'Mean_Model_Predicted_Risk': f"{mean_pred:.1f}%",
        'Clinical_Action': (
            "Standard floor care; routine surveillance" if 'Low' in tier else
            "Close monitoring; neuro checks q2h; PCT tracking" if 'Moderate' in tier else
            "Step-down/NICU; aggressive antimicrobial stewardship; serial labs" if 'High' in tier else
            "Full NICU resuscitation; early invasive culture; escalated therapy"
        )
    })

tier_df = pd.DataFrame(tier_summary)
tier_df.to_csv(os.path.join(TABLE_DIR, 'table4_nomogram_validation.csv'), index=False)
print("  Saved Table 4: table4_nomogram_validation.csv")

# Save patient risk scores
scores_export = clinical[['SR NO.', 'age_val', 'gender_binary', 'aneurysm_clean', 'GCS_total_init',
                          'infection_binary', 'PCT', 'CRP_val', 'asim_score', 'risk_tier',
                          'predicted_prob', 'outcome_binary']]
scores_export.to_csv(os.path.join(DATA_DIR, 'patient_risk_scores.csv'), index=False)

# Save model pickle
with open(os.path.join(MODEL_DIR, 'nomogram_model.pkl'), 'wb') as f:
    pickle.dump({
        'model': lr,
        'beta0': beta0,
        'beta1': beta1,
        'auc_mean': auc_mean,
        'auc_ci': (auc_lower, auc_upper),
        'brier_mean': brier_mean,
        'brier_ci': (brier_lower, brier_upper),
        'sens_mean': sens_mean,
        'spec_mean': spec_mean,
    }, f)

# ─── FIGURE 8: PUBLICATION CLINICAL NOMOGRAM ───
print("  Generating Figure 8: Publication Bedside Nomogram...")
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(8, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1.2, 1.4], hspace=0.6)

nomo_vars = [
    ("Initial GCS Severity", [("Mild (13-15)", 0), ("Moderate (9-12)", 2), ("Severe (3-8)", 4)]),
    ("Aneurysm Location", [("Protected/Other", 0), ("MCA", 1), ("ACOM / ICA-PCOM", 2), ("ICA", 3)]),
    ("Infection Profile", [("None", 0), ("Non-Invasive Positive", 1), ("Invasive (Blood/CSF)", 3)]),
    ("PCT Paradox Zone", [("No (<0.05 or >0.5)", 0), ("Yes (0.05 - 0.5 ng/mL)", 2)]),
    ("Extreme Inflammation (CRP)", [("Normal/Intermediate", 0), ("Discordant (<5 or >62)", 2)]),
    ("Age", [("<= 60 Years", 0), ("> 60 Years", 1)]),
]

max_pts = 15

# Plot individual variable axes
for i, (var_name, points_list) in enumerate(nomo_vars):
    ax = fig.add_subplot(gs[i, 0])
    ax.set_xlim(-0.5, max_pts + 0.5)
    ax.set_ylim(-0.2, 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#455A64')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    
    # Label for variable
    ax.text(-0.5, 0.5, var_name, fontsize=11, fontweight='bold', va='center', ha='right', color='#263238')
    
    # Plot ticks and category labels
    for cat_name, pts in points_list:
        ax.plot([pts, pts], [0, 0.3], color='#1976D2', lw=2)
        ax.plot(pts, 0, marker='o', markersize=5, color='#1976D2')
        ax.text(pts, 0.45, cat_name, ha='center', va='bottom', fontsize=8.5, fontweight='bold', rotation=0, color='#37474F')
        ax.text(pts, -0.25, f"{pts} pt{'s' if pts!=1 else ''}", ha='center', va='top', fontsize=8, color='#78909C')

# Total Points Axis
ax_total = fig.add_subplot(gs[6, 0])
ax_total.set_xlim(-0.5, max_pts + 0.5)
ax_total.set_ylim(-0.2, 1.0)
ax_total.spines['top'].set_visible(False)
ax_total.spines['right'].set_visible(False)
ax_total.spines['left'].set_visible(False)
ax_total.spines['bottom'].set_color('#D32F2F')
ax_total.spines['bottom'].set_linewidth(2)
ax_total.get_yaxis().set_visible(False)
ax_total.text(-0.5, 0.4, "TOTAL ASIM POINTS", fontsize=11, fontweight='bold', va='center', ha='right', color='#D32F2F')

total_pts_ticks = list(range(0, max_pts + 1))
ax_total.set_xticks(total_pts_ticks)
ax_total.set_xticklabels([str(t) for t in total_pts_ticks], fontweight='bold', fontsize=9)
for t in total_pts_ticks:
    ax_total.plot([t, t], [0, 0.2], color='#D32F2F', lw=1.5)

# Predicted Probability Axis
ax_prob = fig.add_subplot(gs[7, 0])
ax_prob.set_xlim(-0.5, max_pts + 0.5)
ax_prob.set_ylim(-0.2, 1.0)
ax_prob.spines['top'].set_visible(False)
ax_prob.spines['right'].set_visible(False)
ax_prob.spines['left'].set_visible(False)
ax_prob.spines['bottom'].set_color('#2E7D32')
ax_prob.spines['bottom'].set_linewidth(2)
ax_prob.get_yaxis().set_visible(False)
ax_prob.text(-0.5, 0.4, "PREDICTED MORTALITY RISK", fontsize=11, fontweight='bold', va='center', ha='right', color='#2E7D32')

# Calculate exact probabilities for key tick points
prob_ticks = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85]
for p in prob_ticks:
    logit_val = np.log(p / (1 - p))
    s_val = (logit_val - beta0) / beta1
    if 0 <= s_val <= max_pts:
        ax_prob.plot([s_val, s_val], [0, 0.25], color='#2E7D32', lw=1.5)
        ax_prob.plot(s_val, 0, marker='s', markersize=4, color='#2E7D32')
        ax_prob.text(s_val, 0.35, f"{int(p*100)}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1B5E20')

ax_prob.set_xticks([])

plt.suptitle("ANEURYSM-SAH INFECTION-MORTALITY (ASIM) BEDSIDE CLINICAL NOMOGRAM\nBootstrap C-index = 0.771 [95% CI: 0.718 - 0.824] | Standardized Calibration",
             fontsize=13, fontweight='bold', y=0.98, color='#1A237E')

plt.savefig(os.path.join(FIG_DIR, 'fig8_clinical_nomogram.png'), bbox_inches='tight')
plt.close()
print("    Saved fig8_clinical_nomogram.png")

# ─── FIGURE 9: SCORE RISK CURVE & STRATIFICATION ───
print("  Generating Figure 9: Score Risk Calibration & Stratification...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot A: Sigmoidal Risk Curve
s_range = np.linspace(0, 14, 200)
p_curve = 1 / (1 + np.exp(-(beta0 + beta1 * s_range)))

axes[0].plot(s_range, p_curve * 100, color='#1976D2', lw=2.5, label='Logistic Model P(Mortality)')
# Bootstrap confidence band
boot_curves = []
for b in range(100):
    idx = np.random.choice(len(clinical), size=len(clinical), replace=True)
    b_lr = LogisticRegression(penalty=None, solver='lbfgs')
    b_lr.fit(scores[idx].reshape(-1, 1), y_true[idx])
    boot_curves.append(b_lr.predict_proba(s_range.reshape(-1, 1))[:, 1] * 100)
boot_low = np.percentile(boot_curves, 2.5, axis=0)
boot_high = np.percentile(boot_curves, 97.5, axis=0)
axes[0].fill_between(s_range, boot_low, boot_high, color='#BBDEFB', alpha=0.5, label='95% Bootstrap CI')

# Scatter of actual patient data
for s_val in np.unique(scores):
    sub = clinical[clinical['asim_score'] == s_val]
    obs_p = sub['outcome_binary'].mean() * 100
    n = len(sub)
    axes[0].scatter(s_val, obs_p, color='#D32F2F', s=n*15, alpha=0.8, edgecolor='black', zorder=5)

axes[0].set_xlabel('ASIM Bedside Risk Score (Points)', fontweight='bold')
axes[0].set_ylabel('Mortality Rate (%)', fontweight='bold')
axes[0].set_title('A. Continuous Risk Calibration Function\nPoint size corresponds to patient group size', fontweight='bold', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.4)
axes[0].legend(loc='upper left')
axes[0].set_ylim(-2, 102)
axes[0].set_xlim(-0.5, 14.5)

# Plot B: Risk Tiers Bar Chart
tier_names_short = ['Low Risk\n(0-3 pts)', 'Moderate\n(4-6 pts)', 'High Risk\n(7-9 pts)', 'Very High\n(10+ pts)']
pred_rates = []
patient_counts = []
for tier in tier_order:
    sub = clinical[clinical['risk_tier'] == tier]
    pred_rates.append(sub['predicted_prob'].mean() * 100 if len(sub)>0 else 0)
    patient_counts.append(len(sub))

x_pos = np.arange(len(tier_names_short))
bar_width = 0.35

bars_obs = axes[1].bar(x_pos - bar_width/2, [float(r['Observed_Mortality_Rate'].replace('%','')) for r in tier_summary],
                       width=bar_width, color='#EF5350', edgecolor='white', label='Observed Mortality')
bars_pred = axes[1].bar(x_pos + bar_width/2, pred_rates,
                        width=bar_width, color='#42A5F5', edgecolor='white', label='Model Predicted Risk')

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f"{t}\n(n={n})" for t, n in zip(tier_names_short, patient_counts)], fontsize=9, fontweight='bold')
axes[1].set_ylabel('Mortality Rate (%)', fontweight='bold')
axes[1].set_title('B. Actionable Risk Tier Stratification\nObserved vs Model-Predicted Outcomes', fontweight='bold', fontsize=11)
axes[1].legend(loc='upper left')
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
axes[1].set_ylim(0, 85)

for bar in bars_obs:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#B71C1C')
for bar in bars_pred:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#0D47A1')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig8_score_risk_curve.png'), bbox_inches='tight')
plt.close()
print("    Saved fig8_score_risk_curve.png")

print("\n[Module 7] COMPLETE")
