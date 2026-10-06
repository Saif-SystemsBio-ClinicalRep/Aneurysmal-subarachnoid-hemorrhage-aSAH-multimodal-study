"""
Module 4: Statistical Analysis & Publication Table 1
=====================================================
Generates publication-ready Table 1 (Survivors vs Deaths) with
appropriate statistical tests, and Table 2 (Aneurysm-stratified).
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = r"C:\Users\SAIF\.gemini\antigravity-ide\scratch\sah_aneurysm_study"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TABLE_DIR = os.path.join(PROJECT_DIR, "output", "tables")

print("[Module 4] Generating publication tables...")

# Load data
unified = pd.read_csv(os.path.join(DATA_DIR, 'unified_feature_matrix.csv'))
suppl = pd.read_csv(os.path.join(DATA_DIR, 'supplementary_vars.csv'))
clinical = pd.read_csv(os.path.join(DATA_DIR, 'clinical_backbone.csv'))
# Merge supplementary (aneurysm_clean, delta_GCS, raw biomarkers)
df = unified.merge(suppl[['SR NO.', 'aneurysm_clean', 'delta_GCS', 'GCS_deteriorated']], on='SR NO.', how='left', suffixes=('', '_sup'))
df = df.merge(clinical[['SR NO.', 'PCT', 'CRP_val', 'TLC_val']], on='SR NO.', how='left', suffixes=('', '_clin'))

# ─── TABLE 1: BASELINE CHARACTERISTICS (SURVIVORS vs DEATHS) ───
print("  Generating Table 1: Survivors vs Deaths...")

survivors = df[df['outcome_binary'] == 0]
deaths = df[df['outcome_binary'] == 1]

def format_continuous(data_surv, data_dead, var_name, display_name):
    """Format continuous variable with median [IQR] and Mann-Whitney U test."""
    s = data_surv[var_name].dropna()
    d = data_dead[var_name].dropna()
    
    s_med = s.median()
    s_q1 = s.quantile(0.25)
    s_q3 = s.quantile(0.75)
    
    d_med = d.median()
    d_q1 = d.quantile(0.25)
    d_q3 = d.quantile(0.75)
    
    if len(s) > 0 and len(d) > 0:
        u_stat, p_val = stats.mannwhitneyu(s, d, alternative='two-sided')
    else:
        u_stat, p_val = np.nan, np.nan
    
    sig = ''
    if p_val < 0.001: sig = '***'
    elif p_val < 0.01: sig = '**'
    elif p_val < 0.05: sig = '*'
    
    return {
        'Variable': display_name,
        'Survivors (n=78)': f"{s_med:.1f} [{s_q1:.1f}-{s_q3:.1f}]",
        'Deaths (n=14)': f"{d_med:.1f} [{d_q1:.1f}-{d_q3:.1f}]",
        'Test': 'Mann-Whitney U',
        'p-value': f"{p_val:.4f}",
        'Sig': sig
    }

def format_categorical(data_all, var_name, display_name, categories=None):
    """Format categorical variable with n (%) and Fisher's exact test."""
    rows = []
    if categories is None:
        categories = sorted(data_all[var_name].unique())
    
    for cat in categories:
        s_n = (survivors[var_name] == cat).sum()
        s_pct = s_n / len(survivors) * 100
        d_n = (deaths[var_name] == cat).sum()
        d_pct = d_n / len(deaths) * 100
        
        rows.append({
            'Variable': f"  {cat}",
            'Survivors (n=78)': f"{s_n} ({s_pct:.1f}%)",
            'Deaths (n=14)': f"{d_n} ({d_pct:.1f}%)",
            'Test': '',
            'p-value': '',
            'Sig': ''
        })
    
    return rows

def format_binary(data_surv, data_dead, var_name, display_name):
    """Format binary variable with n (%) and Fisher's exact test."""
    s_pos = int(data_surv[var_name].sum())
    d_pos = int(data_dead[var_name].sum())
    s_neg = len(data_surv) - s_pos
    d_neg = len(data_dead) - d_pos
    
    table = np.array([[d_pos, d_neg], [s_pos, s_neg]])
    try:
        _, p_val = stats.fisher_exact(table)
    except:
        p_val = np.nan
    
    sig = ''
    if p_val < 0.001: sig = '***'
    elif p_val < 0.01: sig = '**'
    elif p_val < 0.05: sig = '*'
    
    return {
        'Variable': display_name,
        'Survivors (n=78)': f"{s_pos} ({s_pos/len(data_surv)*100:.1f}%)",
        'Deaths (n=14)': f"{d_pos} ({d_pos/len(data_dead)*100:.1f}%)",
        'Test': "Fisher's exact",
        'p-value': f"{p_val:.4f}" if not np.isnan(p_val) else 'N/A',
        'Sig': sig
    }

table1_rows = []

# Demographics
table1_rows.append({'Variable': 'DEMOGRAPHICS', 'Survivors (n=78)': '', 'Deaths (n=14)': '', 'Test': '', 'p-value': '', 'Sig': ''})
table1_rows.append(format_continuous(survivors, deaths, 'age_val', 'Age (years)'))
table1_rows.append(format_binary(survivors, deaths, 'gender_binary', 'Male sex'))

# Aneurysm
table1_rows.append({'Variable': 'ANEURYSM TYPE', 'Survivors (n=78)': '', 'Deaths (n=14)': '', 'Test': '', 'p-value': '', 'Sig': ''})
table1_rows.extend(format_categorical(df, 'aneurysm_clean', 'Aneurysm Type',
    categories=['ACOM', 'ICA-PCOM', 'ICA', 'MCA', 'DACA_ACA', 'Angionegative', 'Basilar_VB']))
table1_rows.append(format_binary(survivors, deaths, 'anterior_circulation', 'Anterior circulation'))

# Clinical
table1_rows.append({'Variable': 'CLINICAL FEATURES', 'Survivors (n=78)': '', 'Deaths (n=14)': '', 'Test': '', 'p-value': '', 'Sig': ''})
table1_rows.append(format_continuous(survivors, deaths, 'GCS_total_init', 'Initial GCS total'))
table1_rows.append(format_continuous(survivors, deaths, 'GCS_M_init', 'Initial GCS Motor'))
table1_rows.append(format_binary(survivors, deaths, 'intubated_init', 'Intubated at admission'))
table1_rows.append(format_continuous(survivors, deaths, 'hospital_stay', 'Hospital stay (days)'))

# GCS severity
table1_rows.append({'Variable': 'GCS SEVERITY', 'Survivors (n=78)': '', 'Deaths (n=14)': '', 'Test': '', 'p-value': '', 'Sig': ''})
for cls, label in [(3, 'Severe (3-8)'), (2, 'Moderate (9-12)'), (1, 'Mild (13-15)')]:
    s_n = (survivors['GCS_severity_class'] == cls).sum()
    d_n = (deaths['GCS_severity_class'] == cls).sum()
    table1_rows.append({
        'Variable': f"  {label}",
        'Survivors (n=78)': f"{s_n} ({s_n/len(survivors)*100:.1f}%)",
        'Deaths (n=14)': f"{d_n} ({d_n/len(deaths)*100:.1f}%)",
        'Test': '', 'p-value': '', 'Sig': ''
    })

# Biomarkers
table1_rows.append({'Variable': 'INFLAMMATORY BIOMARKERS', 'Survivors (n=78)': '', 'Deaths (n=14)': '', 'Test': '', 'p-value': '', 'Sig': ''})
table1_rows.append(format_continuous(survivors, deaths, 'PCT', 'Procalcitonin (ng/mL)'))
table1_rows.append(format_continuous(survivors, deaths, 'CRP_val', 'CRP (mg/L)'))
table1_rows.append(format_continuous(survivors, deaths, 'TLC_val', 'TLC (x10^3/uL)'))
table1_rows.append(format_binary(survivors, deaths, 'PCT_paradox_zone', 'PCT Paradox Zone (0.05-0.5)'))
table1_rows.append(format_binary(survivors, deaths, 'CRP_extreme', 'CRP Extreme (<Q1 or >Q3)'))

# Novel composite indices
table1_rows.append({'Variable': 'NOVEL COMPOSITE INDICES', 'Survivors (n=78)': '', 'Deaths (n=14)': '', 'Test': '', 'p-value': '', 'Sig': ''})
table1_rows.append(format_continuous(survivors, deaths, 'IBI', 'Inflammatory Burden Index'))
table1_rows.append(format_continuous(survivors, deaths, 'log_PCT', 'log10(PCT + 0.01)'))
table1_rows.append(format_continuous(survivors, deaths, 'log_CRP', 'log10(CRP + 1)'))
table1_rows.append(format_continuous(survivors, deaths, 'neural_inflammatory_coupling', 'Neural-Inflammatory Coupling'))

# Infection & Microbiology
table1_rows.append({'Variable': 'INFECTION & MICROBIOLOGY', 'Survivors (n=78)': '', 'Deaths (n=14)': '', 'Test': '', 'p-value': '', 'Sig': ''})
table1_rows.append(format_binary(survivors, deaths, 'infection_binary', 'Nosocomial infection'))
table1_rows.append(format_continuous(survivors, deaths, 'positive_site_count', 'Positive culture sites'))
table1_rows.append(format_binary(survivors, deaths, 'has_MDR', 'MDR organism'))
table1_rows.append(format_binary(survivors, deaths, 'has_carbapenem_resistance', 'Carbapenem resistance'))
table1_rows.append(format_binary(survivors, deaths, 'has_invasive_infection', 'Invasive infection (Blood/CSF)'))
table1_rows.append(format_binary(survivors, deaths, 'is_polymicrobial', 'Polymicrobial infection'))

# Build Table 1 DataFrame
table1_df = pd.DataFrame(table1_rows)
table1_df.to_csv(os.path.join(TABLE_DIR, 'table1_baseline_characteristics.csv'), index=False)

# Print formatted table
print("\n  TABLE 1: Baseline Characteristics")
print("  " + "=" * 100)
for _, row in table1_df.iterrows():
    if row['Test'] == '' and row['Survivors (n=78)'] == '':
        print(f"\n  {row['Variable']}")
    else:
        p_str = f"p={row['p-value']}" if row['p-value'] else ''
        sig_str = row['Sig'] if row['Sig'] else ''
        print(f"    {row['Variable']:<40} {row['Survivors (n=78)']:<25} {row['Deaths (n=14)']:<25} {p_str} {sig_str}")

# ─── TABLE 2: ANEURYSM-LOCATION-STRATIFIED PROFILES ───
print("\n  Generating Table 2: Aneurysm-stratified profiles...")

table2_rows = []
for atype in ['ACOM', 'ICA-PCOM', 'ICA', 'MCA', 'DACA_ACA', 'Angionegative', 'Basilar_VB']:
    sub = df[df['aneurysm_clean'] == atype]
    n = len(sub)
    if n == 0:
        continue
    
    deaths_n = int(sub['outcome_binary'].sum())
    infected_n = int(sub['infection_binary'].sum())
    
    surv_sub = sub[sub['outcome_binary'] == 0]
    delta_gcs = surv_sub['delta_GCS'].mean() if len(surv_sub) > 0 else np.nan
    
    table2_rows.append({
        'Aneurysm Type': atype,
        'n': n,
        'Deaths n (%)': f"{deaths_n} ({deaths_n/n*100:.1f}%)",
        'Infected n (%)': f"{infected_n} ({infected_n/n*100:.1f}%)",
        'Age (mean)': f"{sub['age_val'].mean():.1f}",
        'GCS Initial (mean)': f"{sub['GCS_total_init'].mean():.1f}",
        'Stay (mean)': f"{sub['hospital_stay'].mean():.1f}",
        'PCT (median)': f"{sub['PCT'].median():.2f}",
        'CRP (median)': f"{sub['CRP_val'].median():.1f}",
        'Delta GCS (survivors)': f"{delta_gcs:.2f}" if not np.isnan(delta_gcs) else 'N/A',
        'MDR n': int(sub['has_MDR'].sum()),
        'Carbapenem-R n': int(sub['has_carbapenem_resistance'].sum()),
    })

table2_df = pd.DataFrame(table2_rows)
table2_df.to_csv(os.path.join(TABLE_DIR, 'table2_aneurysm_stratified.csv'), index=False)

print("\n  TABLE 2: Aneurysm-Location-Stratified Profiles")
print(table2_df.to_string(index=False))

# ─── ODDS RATIOS & KEY ASSOCIATIONS ───
print("\n  Key Statistical Associations:")

# Infection -> Mortality OR
inf_dead = ((df['infection_binary']==1) & (df['outcome_binary']==1)).sum()
inf_live = ((df['infection_binary']==1) & (df['outcome_binary']==0)).sum()
noinf_dead = ((df['infection_binary']==0) & (df['outcome_binary']==1)).sum()
noinf_live = ((df['infection_binary']==0) & (df['outcome_binary']==0)).sum()
table = np.array([[inf_dead, inf_live], [noinf_dead, noinf_live]])
or_val = (inf_dead * noinf_live) / (inf_live * noinf_dead) if (inf_live * noinf_dead) > 0 else np.inf
_, p_fisher = stats.fisher_exact(table)
print(f"    Infection -> Mortality: OR={or_val:.2f}, p={p_fisher:.4f}")

# MDR -> Mortality OR
mdr_dead = ((df['has_MDR']==1) & (df['outcome_binary']==1)).sum()
mdr_live = ((df['has_MDR']==1) & (df['outcome_binary']==0)).sum()
nomdr_dead = ((df['has_MDR']==0) & (df['outcome_binary']==1)).sum()
nomdr_live = ((df['has_MDR']==0) & (df['outcome_binary']==0)).sum()
table2 = np.array([[mdr_dead, mdr_live], [nomdr_dead, nomdr_live]])
if (mdr_live * nomdr_dead) > 0:
    or2 = (mdr_dead * nomdr_live) / (mdr_live * nomdr_dead)
else:
    or2 = np.inf
_, p2 = stats.fisher_exact(table2)
print(f"    MDR -> Mortality: OR={or2:.2f}, p={p2:.4f}")

# Invasive infection -> Mortality
inv_dead = ((df['has_invasive_infection']==1) & (df['outcome_binary']==1)).sum()
inv_live = ((df['has_invasive_infection']==1) & (df['outcome_binary']==0)).sum()
noinv_dead = ((df['has_invasive_infection']==0) & (df['outcome_binary']==1)).sum()
noinv_live = ((df['has_invasive_infection']==0) & (df['outcome_binary']==0)).sum()
table3 = np.array([[inv_dead, inv_live], [noinv_dead, noinv_live]])
if (inv_live * noinv_dead) > 0:
    or3 = (inv_dead * noinv_live) / (inv_live * noinv_dead)
else:
    or3 = np.inf
_, p3 = stats.fisher_exact(table3)
print(f"    Invasive infection -> Mortality: OR={or3:.2f}, p={p3:.4f}")

print(f"\n[Module 4] COMPLETE")
print(f"  Saved table1_baseline_characteristics.csv")
print(f"  Saved table2_aneurysm_stratified.csv")
