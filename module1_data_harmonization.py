"""
Module 1: Data Harmonization & GCS Intelligence Engine
=======================================================
Parses the master clinical sheet (Sheet4), decomposes GCS strings into
numeric components (E, V, M, Total), standardizes aneurysm types and
outcomes, and prepares the clinical backbone dataframe.
"""

import pandas as pd
import numpy as np
import re
import os
import sys

# ─── PATHS ───
XLSX_PATH = r"C:\Users\SAIF\Downloads\predictive model (1).xlsx"
PROJECT_DIR = r"C:\Users\SAIF\.gemini\antigravity-ide\scratch\sah_aneurysm_study"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ─── 1. LOAD MASTER SHEET ───
print("[Module 1] Loading master clinical sheet...")
df = pd.read_excel(XLSX_PATH, sheet_name='Sheet4')
df.columns = [str(c).strip() for c in df.columns]
print(f"  Loaded {df.shape[0]} patients, {df.shape[1]} columns")

# ─── 2. DROP PATIENT NAME (DE-IDENTIFICATION) ───
# Store names temporarily for cross-sheet linking (Module 2), then drop
patient_names = df['PATIENT NAME'].str.strip().tolist()
patient_ids = df['SR NO.'].tolist()
name_to_id = {name: sid for name, sid in zip(patient_names, patient_ids)}

# ─── 3. STANDARDIZE OUTCOME ───
df['outcome_binary'] = df['outcome'].str.strip().str.upper().map({'DEATH': 1, 'LIVE': 0})
assert df['outcome_binary'].isna().sum() == 0, "Unmapped outcome values found!"
print(f"  Outcomes: Survived={int((df['outcome_binary']==0).sum())}, Deaths={int((df['outcome_binary']==1).sum())}")

# ─── 4. STANDARDIZE INFECTION STATUS ───
df['infection_binary'] = df['Infection'].str.strip().str.upper().apply(lambda x: 1 if x == 'YES' else 0)
print(f"  Infection: Infected={int(df['infection_binary'].sum())}, Non-infected={int((df['infection_binary']==0).sum())}")

# ─── 5. STANDARDIZE GENDER ───
df['gender_binary'] = df['gender'].str.strip().str.upper().map({'M': 1, 'F': 0})
print(f"  Gender: Male={int(df['gender_binary'].sum())}, Female={int((df['gender_binary']==0).sum())}")

# ─── 6. STANDARDIZE ANEURYSM TYPE ───
def clean_aneurysm(x):
    s = str(x).strip().upper()
    if 'PCOM' in s:
        return 'ICA-PCOM'
    if 'ACOM' in s:
        return 'ACOM'
    if 'MCA' in s:
        return 'MCA'
    if 'DACA' in s or ('ACA' in s and 'ICA' not in s):
        return 'DACA_ACA'
    if 'BASILAR' in s or 'VERTEBRO' in s:
        return 'Basilar_VB'
    if 'ANGIO' in s:
        return 'Angionegative'
    if 'ICA' in s:
        return 'ICA'
    return 'Other'

df['aneurysm_clean'] = df['Aneurysm Type'].apply(clean_aneurysm)
print(f"  Aneurysm types: {df['aneurysm_clean'].value_counts().to_dict()}")

# Anterior vs Posterior circulation
df['anterior_circulation'] = df['aneurysm_clean'].apply(
    lambda a: 1 if a in ['ACOM', 'ICA-PCOM', 'ICA', 'MCA', 'DACA_ACA'] else 0
)

# High-risk location (locations with observed deaths)
df['high_risk_location'] = df['aneurysm_clean'].apply(
    lambda a: 1 if a in ['ICA', 'ICA-PCOM', 'ACOM', 'MCA'] else 0
)

# Location-specific empirical mortality weight
location_mort = {'ICA': 0.267, 'ICA-PCOM': 0.176, 'ACOM': 0.162, 'MCA': 0.125,
                 'DACA_ACA': 0.0, 'Angionegative': 0.0, 'Basilar_VB': 0.0}
df['location_mortality_weight'] = df['aneurysm_clean'].map(location_mort).fillna(0)

# Location-specific empirical infection weight
location_inf = {'Basilar_VB': 0.667, 'ACOM': 0.324, 'MCA': 0.250, 'ICA-PCOM': 0.235,
                'ICA': 0.200, 'DACA_ACA': 0.167, 'Angionegative': 0.0}
df['location_infection_weight'] = df['aneurysm_clean'].map(location_inf).fillna(0)

# One-hot encode aneurysm type
for atype in ['ACOM', 'ICA-PCOM', 'ICA', 'MCA']:
    df[f'aneurysm_{atype.replace("-","_")}'] = (df['aneurysm_clean'] == atype).astype(int)
df['aneurysm_Other'] = df['aneurysm_clean'].apply(
    lambda x: 1 if x in ['DACA_ACA', 'Angionegative', 'Basilar_VB'] else 0
)

# ─── 7. PARSE GCS STRINGS ───
def parse_gcs(gcs_str):
    """
    Parse GCS string like E4V5M6 or E1VTM5 into numeric components.
    Returns: (E, V, M, Total, Intubated)
    """
    if pd.isna(gcs_str):
        return pd.Series([np.nan, np.nan, np.nan, np.nan, 0])
    s = str(gcs_str).strip().upper()
    if s == 'DEATH':
        return pd.Series([1.0, 1.0, 1.0, 3.0, 0])
    m = re.match(r'E(\d)V(T|\d)M(\d)', s)
    if not m:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, 0])
    e = float(m.group(1))
    intubated = 1 if m.group(2) == 'T' else 0
    v = 1.0 if m.group(2) == 'T' else float(m.group(2))
    mo = float(m.group(3))
    total = e + v + mo
    return pd.Series([e, v, mo, total, intubated])

# Parse Initial GCS
df[['GCS_E_init', 'GCS_V_init', 'GCS_M_init', 'GCS_total_init', 'intubated_init']] = \
    df['initial gcs'].apply(parse_gcs)

# Parse Discharge GCS
df[['GCS_E_disc', 'GCS_V_disc', 'GCS_M_disc', 'GCS_total_disc', 'intubated_disc']] = \
    df['discharge gcs'].apply(parse_gcs)

print(f"  GCS Initial: mean={df['GCS_total_init'].mean():.1f}, range=[{df['GCS_total_init'].min():.0f}-{df['GCS_total_init'].max():.0f}]")
print(f"  Intubated at admission: {int(df['intubated_init'].sum())}")

# Delta GCS
df['delta_GCS'] = df['GCS_total_disc'] - df['GCS_total_init']
df['GCS_improved'] = (df['delta_GCS'] > 0).astype(int)
df['GCS_deteriorated'] = ((df['delta_GCS'] < 0) | (df['outcome_binary'] == 1)).astype(int)

# GCS severity classification
def gcs_severity(g):
    if pd.isna(g):
        return np.nan
    if g <= 8:
        return 3  # Severe
    elif g <= 12:
        return 2  # Moderate
    else:
        return 1  # Mild
df['GCS_severity_class'] = df['GCS_total_init'].apply(gcs_severity)

print(f"  GCS severity: Severe(3-8)={int((df['GCS_severity_class']==3).sum())}, "
      f"Moderate(9-12)={int((df['GCS_severity_class']==2).sum())}, "
      f"Mild(13-15)={int((df['GCS_severity_class']==1).sum())}")

# ─── 8. CLEAN NUMERIC BIOMARKERS ───
df['PCT'] = pd.to_numeric(df['Procal'], errors='coerce')
df['CRP_val'] = pd.to_numeric(df['CRP'], errors='coerce')
df['TLC_val'] = pd.to_numeric(df['TLC'], errors='coerce')
df['hospital_stay'] = pd.to_numeric(df['stay'], errors='coerce')
df['age_val'] = pd.to_numeric(df['age'], errors='coerce')

print(f"  PCT: mean={df['PCT'].mean():.3f}, median={df['PCT'].median():.3f}, zeros={int((df['PCT']==0).sum())}")
print(f"  CRP: mean={df['CRP_val'].mean():.1f}, median={df['CRP_val'].median():.1f}, zeros={int((df['CRP_val']==0).sum())}")
print(f"  TLC: mean={df['TLC_val'].mean():.1f}, range=[{df['TLC_val'].min():.1f}-{df['TLC_val'].max():.1f}]")

# ─── 9. EXTRACT CULTURE SITE DATA ───
culture_cols_map = {
    'Urine': 'urine',
    'Blood': 'blood',
    'Sputum': 'sputum',
    'CVP': 'cvp_line',
    'CSF': 'csf',
    'Tracheal aspirate': 'tracheal',
    'Pus': 'pus'
}

df['positive_site_count'] = 0
for col_name, site_key in culture_cols_map.items():
    if col_name in df.columns:
        positive = df[col_name].notna() & (df[col_name].astype(str).str.strip() != '')
        df[f'culture_{site_key}_positive'] = positive.astype(int)
        df['positive_site_count'] += positive.astype(int)

print(f"  Positive site distribution: {df['positive_site_count'].value_counts().sort_index().to_dict()}")

# ─── 10. SAVE INTERMEDIATE OUTPUTS ───
name_map_df = pd.DataFrame({
    'SR_NO': patient_ids,
    'PATIENT_NAME': patient_names
})
name_map_df.to_csv(os.path.join(DATA_DIR, 'patient_name_map.csv'), index=False)

clinical_cols = [
    'SR NO.', 'age_val', 'gender_binary', 'aneurysm_clean',
    'anterior_circulation', 'high_risk_location',
    'location_mortality_weight', 'location_infection_weight',
    'aneurysm_ACOM', 'aneurysm_ICA_PCOM', 'aneurysm_ICA', 'aneurysm_MCA', 'aneurysm_Other',
    'GCS_E_init', 'GCS_V_init', 'GCS_M_init', 'GCS_total_init', 'intubated_init',
    'GCS_E_disc', 'GCS_V_disc', 'GCS_M_disc', 'GCS_total_disc', 'intubated_disc',
    'delta_GCS', 'GCS_improved', 'GCS_deteriorated', 'GCS_severity_class',
    'hospital_stay', 'PCT', 'CRP_val', 'TLC_val',
    'infection_binary', 'positive_site_count',
    'culture_urine_positive', 'culture_blood_positive', 'culture_sputum_positive',
    'culture_cvp_line_positive', 'culture_csf_positive', 'culture_tracheal_positive',
    'culture_pus_positive',
    'outcome_binary'
]

clinical_cols = [c for c in clinical_cols if c in df.columns]
df_clinical = df[clinical_cols].copy()
df_clinical.to_csv(os.path.join(DATA_DIR, 'clinical_backbone.csv'), index=False)

print(f"\n[Module 1] COMPLETE")
print(f"  Saved clinical_backbone.csv ({df_clinical.shape[0]} rows, {df_clinical.shape[1]} columns)")
print(f"  Saved patient_name_map.csv ({len(patient_ids)} entries)")
