"""
Module 3: Feature Engineering & Unified Feature Matrix
========================================================
Merges clinical backbone + resistome features, computes all novel
composite biomarker indices and interaction features. Outputs the
final analysis-ready feature matrix.
"""

import pandas as pd
import numpy as np
import os

PROJECT_DIR = r"C:\Users\SAIF\.gemini\antigravity-ide\scratch\sah_aneurysm_study"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

print("[Module 3] Building unified feature matrix...")

# ─── LOAD COMPONENTS ───
clinical = pd.read_csv(os.path.join(DATA_DIR, 'clinical_backbone.csv'))
resistome = pd.read_csv(os.path.join(DATA_DIR, 'resistome_features.csv'))

# Merge on SR NO.
df = clinical.merge(resistome.drop(columns=['SR NO.'], errors='ignore'),
                    left_index=True, right_index=True, how='left')
print(f"  Merged dataframe: {df.shape[0]} rows, {df.shape[1]} columns")

# ─── COMPOSITE BIOMARKER INDICES ───
print("  Computing composite biomarker indices...")

# 1. Log-transformed biomarkers (handles zero-inflation and right-skew)
df['log_PCT'] = np.log10(df['PCT'] + 0.01)
df['log_CRP'] = np.log10(df['CRP_val'] + 1.0)
df['TLC_norm'] = df['TLC_val'] / 10.0

# 2. Inflammatory Burden Index (IBI) - Novel multiplicative composite
df['IBI'] = df['log_CRP'] * np.abs(df['log_PCT']) * df['TLC_norm']

# 3. PCT-to-CRP Discrimination Ratio - Novel
df['PCT_CRP_ratio'] = df['PCT'] / (df['CRP_val'] + 0.1)

# 4. CRP Extreme flag (U-shaped relationship with mortality)
crp_q1 = df['CRP_val'].quantile(0.25)
crp_q3 = df['CRP_val'].quantile(0.75)
df['CRP_extreme'] = ((df['CRP_val'] < crp_q1) | (df['CRP_val'] > crp_q3)).astype(int)

# 5. Neural-Inflammatory Coupling - Novel
df['neural_inflammatory_coupling'] = (15 - df['GCS_total_init']) * df['log_CRP']

# 6. PCT Paradox Zone Flag - Novel (borderline PCT carries highest mortality)
df['PCT_paradox_zone'] = ((df['PCT'] >= 0.05) & (df['PCT'] <= 0.5)).astype(int)

# 7. Immunoparalysis Index - Novel
df['immunoparalysis'] = ((df['CRP_val'] < 5) & (df['PCT'] < 0.05) & (df['infection_binary'] == 1)).astype(int)

# ─── INTERACTION FEATURES ───
print("  Computing interaction features...")

# Age-GCS risk interaction
df['age_GCS_risk'] = df['age_val'] * (15 - df['GCS_total_init'])

# Location-GCS interaction
df['location_GCS_interaction'] = df['location_mortality_weight'] * (15 - df['GCS_total_init'])

# Stay-infection interaction
df['stay_infection_product'] = df['hospital_stay'] * df['infection_binary']

# Location-infection synergy
df['location_infection_synergy'] = df['location_infection_weight'] * df['infection_binary']

# ACOM-specific infection interaction (ACOM deaths are infection-mediated)
df['ACOM_infected'] = df['aneurysm_ACOM'] * df['infection_binary']

# ICA-specific low GCS interaction (ICA deaths are GCS-mediated)
df['ICA_lowGCS'] = df['aneurysm_ICA'] * (df['GCS_total_init'] <= 12).astype(int)

# ─── BUILD FINAL FEATURE MATRIX ───
print("  Building final feature matrix...")

# Define the complete feature set
feature_columns = [
    # Demographics
    'age_val', 'gender_binary',
    # Aneurysm features
    'aneurysm_ACOM', 'aneurysm_ICA_PCOM', 'aneurysm_ICA', 'aneurysm_MCA', 'aneurysm_Other',
    'anterior_circulation', 'high_risk_location',
    'location_mortality_weight', 'location_infection_weight',
    # GCS features
    'GCS_total_init', 'GCS_M_init', 'GCS_E_init',
    'intubated_init', 'GCS_severity_class',
    # Hospital stay
    'hospital_stay',
    # Raw biomarkers (log-transformed)
    'log_PCT', 'log_CRP', 'TLC_norm',
    # Novel composite indices
    'IBI', 'PCT_CRP_ratio', 'CRP_extreme',
    'neural_inflammatory_coupling', 'PCT_paradox_zone', 'immunoparalysis',
    # Infection features
    'infection_binary', 'positive_site_count',
    # Resistome features
    'total_organisms', 'gram_negative_count', 'is_polymicrobial',
    'has_MDR', 'has_carbapenem_resistance',
    'has_bloodstream_infection', 'has_CSF_infection', 'has_respiratory_infection',
    'has_invasive_infection', 'ARI_score',
    # Interaction features
    'age_GCS_risk', 'location_GCS_interaction',
    'stay_infection_product', 'location_infection_synergy',
    'ACOM_infected', 'ICA_lowGCS',
]

# Ensure all columns exist
feature_columns = [c for c in feature_columns if c in df.columns]
target_col = 'outcome_binary'

# Build matrices
X = df[feature_columns].copy()
y = df[target_col].copy()

# Fill any remaining NaN with 0 for resistome features (non-infected patients)
X = X.fillna(0)

# Replace infinity values
X = X.replace([np.inf, -np.inf], 0)

print(f"  Feature matrix shape: {X.shape}")
print(f"  Target distribution: {y.value_counts().to_dict()}")
print(f"  Features ({len(feature_columns)}):")
for i, f in enumerate(feature_columns):
    print(f"    [{i+1}] {f}: mean={X[f].mean():.3f}, std={X[f].std():.3f}")

# Save the unified feature matrix
unified = pd.concat([df[['SR NO.']], X, y], axis=1)
unified.to_csv(os.path.join(DATA_DIR, 'unified_feature_matrix.csv'), index=False)

# Save feature names
with open(os.path.join(DATA_DIR, 'feature_names.txt'), 'w') as f:
    for feat in feature_columns:
        f.write(feat + '\n')

# Also save supplementary columns for analysis (aneurysm_clean for stratification)
df[['SR NO.', 'aneurysm_clean', 'delta_GCS', 'GCS_deteriorated',
    'PCT', 'CRP_val', 'TLC_val', 'GCS_total_init', 'GCS_total_disc']].to_csv(
    os.path.join(DATA_DIR, 'supplementary_vars.csv'), index=False
)

print(f"\n[Module 3] COMPLETE")
print(f"  Saved unified_feature_matrix.csv ({unified.shape[0]} rows, {unified.shape[1]} columns)")
print(f"  Total engineered features: {len(feature_columns)}")
