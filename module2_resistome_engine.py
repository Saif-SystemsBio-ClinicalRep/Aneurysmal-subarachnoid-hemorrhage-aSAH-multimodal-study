"""
Module 2: Resistome Cross-Sheet Linkage Engine
================================================
Links patient names from Sheet4 to their antibiotic susceptibility records
across the 7 AST sheets (Urine, Blood, Sputum, CVP, CSF, Tracheal, Pus).
Standardizes organism taxonomy and antibiotic names, then computes per-patient
resistome features (MDR, carbapenem resistance, ARI, etc.).
"""

import pandas as pd
import numpy as np
import re
import os

XLSX_PATH = r"C:\Users\SAIF\Downloads\predictive model (1).xlsx"
PROJECT_DIR = r"C:\Users\SAIF\.gemini\antigravity-ide\scratch\sah_aneurysm_study"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ─── ORGANISM TAXONOMY STANDARDIZATION ───
ORGANISM_MAP = {
    'k. pneumoniae': 'Klebsiella pneumoniae',
    'k. pneunominae': 'Klebsiella pneumoniae',
    'k. oneumoniae': 'Klebsiella pneumoniae',
    'k.pneumoniae': 'Klebsiella pneumoniae',
    'klebsiella pneumoniae': 'Klebsiella pneumoniae',
    'k. pneumoniae': 'Klebsiella pneumoniae',
    'k. oxytoca': 'Klebsiella oxytoca',
    'klebsiella oxytoca': 'Klebsiella oxytoca',
    'a. baumannii': 'Acinetobacter baumannii',
    'a. baumanni': 'Acinetobacter baumannii',
    'acinetobacter baumannii': 'Acinetobacter baumannii',
    'p. aeruginosa': 'Pseudomonas aeruginosa',
    'pseudomonas aeruginosa': 'Pseudomonas aeruginosa',
    'p. stutzeri': 'Pseudomonas stutzeri',
    'pseudomonas stutzeri': 'Pseudomonas stutzeri',
    'e. coli': 'Escherichia coli',
    'e coli': 'Escherichia coli',
    'e.coli': 'Escherichia coli',
    'enterobacter hormoechei': 'Enterobacter hormaechei',
    'enterobacter aerogenes': 'Enterobacter aerogenes',
    'e. aerogenes': 'Enterobacter aerogenes',
    'providencia rettgeri': 'Providencia rettgeri',
    'citrobacter freundii': 'Citrobacter freundii',
    'c. freundii': 'Citrobacter freundii',
    'c. koseri': 'Citrobacter koseri',
    'citrobacter koseri': 'Citrobacter koseri',
    'paracoccus yeei': 'Paracoccus yeei',
    'enterococcus faecalis': 'Enterococcus faecalis',
    'enterococcus faecium': 'Enterococcus faecium',
    'enterococcus casseliflavus': 'Enterococcus casseliflavus',
    'mrsa': 'MRSA',
    'mssa': 'MSSA',
}

# Gram classification
GRAM_CLASSIFICATION = {
    'Klebsiella pneumoniae': 'Gram-negative',
    'Klebsiella oxytoca': 'Gram-negative',
    'Acinetobacter baumannii': 'Gram-negative',
    'Pseudomonas aeruginosa': 'Gram-negative',
    'Pseudomonas stutzeri': 'Gram-negative',
    'Escherichia coli': 'Gram-negative',
    'Enterobacter hormaechei': 'Gram-negative',
    'Enterobacter aerogenes': 'Gram-negative',
    'Providencia rettgeri': 'Gram-negative',
    'Citrobacter freundii': 'Gram-negative',
    'Citrobacter koseri': 'Gram-negative',
    'Paracoccus yeei': 'Gram-negative',
    'MRSA': 'Gram-positive',
    'MSSA': 'Gram-positive',
    'Enterococcus faecalis': 'Gram-positive',
    'Enterococcus faecium': 'Gram-positive',
    'Enterococcus casseliflavus': 'Gram-positive',
}

# Antibiotic name standardization
ANTIBIOTIC_MAP = {
    'amikacin': 'Amikacin',
    'ceftazidime': 'Ceftazidime',
    'cetazidime': 'Ceftazidime',
    'ceftriaxone': 'Ceftriaxone',
    'ciprofloxacin': 'Ciprofloxacin',
    'colistin': 'Colistin',
    'imipenem': 'Imipenem',
    'meropenem': 'Meropenem',
    'ertapenem': 'Ertapenem',
    'ertapenum': 'Ertapenem',
    'gentamicin': 'Gentamicin',
    'levofloxacin': 'Levofloxacin',
    'nitrofurantoin': 'Nitrofurantoin',
    'norfloxacin': 'Norfloxacin',
    'tazobactum+piperacillin': 'Piperacillin-Tazobactam',
    'trimethoprim-sulfamethoxazole': 'Trimethoprim-Sulfamethoxazole',
    'trimethoprim+ sulfamethoxazole': 'Trimethoprim-Sulfamethoxazole',
    'trimethprim-sulfamethoxazole': 'Trimethoprim-Sulfamethoxazole',
    'sulbactam+cefoperazone': 'Cefoperazone-Sulbactam',
    'sulbactum+cefoperazone': 'Cefoperazone-Sulbactam',
    'sulbactum+cefoprazone': 'Cefoperazone-Sulbactam',
    'minocycline': 'Minocycline',
    'doxycycline': 'Doxycycline',
    'doxicycline': 'Doxycycline',
    'vancomycin': 'Vancomycin',
    'teicoplanin': 'Teicoplanin',
    'linezolid': 'Linezolid',
    'linizolid': 'Linezolid',
    'ampicillin': 'Ampicillin',
    'ampicillin sulbactum': 'Ampicillin-Sulbactam',
    'ampi-sulbactum (as)': 'Ampicillin-Sulbactam',
    'sulbactum+ampicillin': 'Ampicillin-Sulbactam',
    'erythromycin': 'Erythromycin',
    'clindamycin': 'Clindamycin',
    'cefoxitin': 'Cefoxitin',
    'fosfomycin': 'Fosfomycin',
    'ticarcillin': 'Ticarcillin',
    'ticarcillin+ clavulinic acid': 'Ticarcillin-Clavulanate',
    'ticarcillin+clavulanic acid': 'Ticarcillin-Clavulanate',
    'ticarcillin+cluvulanic acid': 'Ticarcillin-Clavulanate',
    'astreonam': 'Aztreonam',
    'aztreonam': 'Aztreonam',
    'levonadifloxacin': 'Levonadifloxacin',
}

# Antibiotic class mapping (for MDR determination)
ANTIBIOTIC_CLASSES = {
    'Amikacin': 'Aminoglycosides',
    'Gentamicin': 'Aminoglycosides',
    'Ceftazidime': 'Cephalosporins',
    'Ceftriaxone': 'Cephalosporins',
    'Cefoperazone-Sulbactam': 'Cephalosporins',
    'Cefoxitin': 'Cephalosporins',
    'Ciprofloxacin': 'Fluoroquinolones',
    'Levofloxacin': 'Fluoroquinolones',
    'Norfloxacin': 'Fluoroquinolones',
    'Levonadifloxacin': 'Fluoroquinolones',
    'Imipenem': 'Carbapenems',
    'Meropenem': 'Carbapenems',
    'Ertapenem': 'Carbapenems',
    'Colistin': 'Polymyxins',
    'Piperacillin-Tazobactam': 'Penicillins',
    'Ampicillin': 'Penicillins',
    'Ampicillin-Sulbactam': 'Penicillins',
    'Ticarcillin': 'Penicillins',
    'Ticarcillin-Clavulanate': 'Penicillins',
    'Trimethoprim-Sulfamethoxazole': 'Folate-pathway-inhibitors',
    'Nitrofurantoin': 'Nitrofurans',
    'Minocycline': 'Tetracyclines',
    'Doxycycline': 'Tetracyclines',
    'Vancomycin': 'Glycopeptides',
    'Teicoplanin': 'Glycopeptides',
    'Linezolid': 'Oxazolidinones',
    'Erythromycin': 'Macrolides',
    'Clindamycin': 'Lincosamides',
    'Fosfomycin': 'Phosphonics',
    'Aztreonam': 'Monobactams',
}

def standardize_organism(org_str):
    """Standardize organism name using the mapping dictionary."""
    s = org_str.strip().lower()
    # Remove leading/trailing parentheses and spaces
    s = re.sub(r'^[\(\s]+|[\)\s]+$', '', s)
    s = s.strip().lower()
    return ORGANISM_MAP.get(s, org_str.strip())

def standardize_antibiotic(abx_str):
    """Standardize antibiotic name."""
    s = abx_str.strip().lower()
    return ANTIBIOTIC_MAP.get(s, abx_str.strip().title())

def standardize_sensitivity(val):
    """Standardize R/S/MS values."""
    s = str(val).strip().upper()
    if s == 'R':
        return 'R'
    elif s == 'S':
        return 'S'
    elif s == 'MS':
        return 'MS'
    else:
        return None  # NIL, empty, NAN, etc.

def extract_patient_name(header_str):
    """Extract patient name from AST sheet header like 'Sita (K. oxytoca) (28/4/25)'."""
    s = str(header_str).strip()
    # Try to extract name before first parenthesis
    m = re.match(r'^([A-Za-z\s\.]+?)[\s]*\(', s)
    if m:
        return m.group(1).strip().upper()
    # Try standalone like "MSSA" or just a name
    m2 = re.match(r'^([A-Za-z\s\.]+)', s)
    if m2:
        name = m2.group(1).strip().upper()
        # Exclude if it's just an organism name
        if name not in ['MRSA', 'MSSA', 'ANTIBIOTIC', 'SENSITIVITY']:
            return name
    return None

def extract_organisms_from_header(header_str):
    """Extract organism names from parentheses in header."""
    orgs = re.findall(r'\(([^)]+)\)', str(header_str))
    result = []
    for o in orgs:
        o_clean = o.strip()
        # Skip dates (contain /)
        if '/' in o_clean and any(c.isdigit() for c in o_clean):
            continue
        # Skip if mostly digits
        if sum(c.isdigit() for c in o_clean) > len(o_clean) / 2:
            continue
        if o_clean:
            result.append(standardize_organism(o_clean))
    # Also handle standalone MRSA/MSSA not in parentheses
    s = str(header_str).strip().upper()
    if 'MRSA' in s and 'MRSA' not in str(result):
        result.append('MRSA')
    if 'MSSA' in s and 'MSSA' not in str(result):
        result.append('MSSA')
    return result

def fuzzy_match_name(name, name_list):
    """Simple fuzzy match: find the closest patient name from the map."""
    name_upper = name.upper().strip()
    # Direct match
    for n in name_list:
        if n.upper().strip() == name_upper:
            return n
    # Partial match (first name match)
    name_parts = name_upper.split()
    for n in name_list:
        n_parts = n.upper().strip().split()
        if name_parts[0] in n_parts or n_parts[0] in name_parts:
            return n
    # Contains match
    for n in name_list:
        if name_upper in n.upper() or n.upper() in name_upper:
            return n
    return None


# ─── LOAD PATIENT NAME MAP ───
print("[Module 2] Loading patient name map...")
name_map = pd.read_csv(os.path.join(DATA_DIR, 'patient_name_map.csv'))
patient_names = name_map['PATIENT_NAME'].tolist()
name_to_id = dict(zip(name_map['PATIENT_NAME'], name_map['SR_NO']))

# ─── PARSE ALL AST SHEETS ───
ast_sheets = {
    'urine': 'urine',
    'blood': 'blood',
    'sputum': 'Sputum',
    'cvp_line': 'CVP line',
    'csf': 'CSF',
    'tracheal': 'Tracheal aspirate',
    'pus': 'PUS'
}

all_ast_records = []  # List of {patient_id, site, organism, gram, antibiotic, sensitivity}

for site_key, sheet_name in ast_sheets.items():
    print(f"\n  Processing {sheet_name} sheet...")
    try:
        ast_df = pd.read_excel(XLSX_PATH, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"    ERROR loading sheet: {e}")
        continue

    # Find the row with organism/patient headers and the rows with antibiotic data
    header_row_idx = None
    abx_start_row = None

    for idx in range(min(10, len(ast_df))):
        row_vals = [str(v).strip() for v in ast_df.iloc[idx].tolist() if pd.notna(v) and str(v).strip()]
        has_org = any('(' in v or 'MRSA' in v.upper() or 'MSSA' in v.upper() for v in row_vals)
        if has_org and header_row_idx is None:
            header_row_idx = idx

    # Parse organism headers from the identified row
    if header_row_idx is None:
        print(f"    WARNING: Could not find organism header row")
        continue

    # Map column index to (patient_name, organism_list)
    col_to_patient_org = {}
    for col_idx in range(1, len(ast_df.columns)):
        val = ast_df.iloc[header_row_idx, col_idx]
        if pd.isna(val) or str(val).strip() == '':
            continue
        header_str = str(val).strip()
        pname = extract_patient_name(header_str)
        orgs = extract_organisms_from_header(header_str)

        if pname and orgs:
            matched_name = fuzzy_match_name(pname, patient_names)
            if matched_name:
                patient_id = name_to_id[matched_name]
                col_to_patient_org[col_idx] = (patient_id, matched_name, orgs)
            else:
                print(f"    WARNING: Could not match patient '{pname}' from header: {header_str}")
                col_to_patient_org[col_idx] = (None, pname, orgs)

    if not col_to_patient_org:
        print(f"    WARNING: No patient-organism mappings found")
        continue

    print(f"    Found {len(col_to_patient_org)} isolate columns")

    # Parse antibiotic sensitivity rows
    for row_idx in range(len(ast_df)):
        abx_name_raw = ast_df.iloc[row_idx, 0]
        if pd.isna(abx_name_raw):
            continue
        abx_raw = str(abx_name_raw).strip()
        if not abx_raw or abx_raw.upper() in ['ANTIBIOTIC', 'SENSITIVITY', '']:
            continue
        # Check if it looks like an antibiotic name (not a patient header)
        if '(' in abx_raw:
            continue

        abx_std = standardize_antibiotic(abx_raw)

        for col_idx, (patient_id, pname, orgs) in col_to_patient_org.items():
            if col_idx >= len(ast_df.columns):
                continue
            sens_val = ast_df.iloc[row_idx, col_idx]
            sens_std = standardize_sensitivity(sens_val)
            if sens_std is None:
                continue

            # Each organism in the header gets this sensitivity result
            for org in orgs:
                gram = GRAM_CLASSIFICATION.get(org, 'Unknown')
                abx_class = ANTIBIOTIC_CLASSES.get(abx_std, 'Other')
                all_ast_records.append({
                    'patient_id': patient_id,
                    'patient_name': pname,
                    'site': site_key,
                    'organism': org,
                    'gram_stain': gram,
                    'antibiotic': abx_std,
                    'antibiotic_class': abx_class,
                    'sensitivity': sens_std,
                })

# ─── BUILD AST DATAFRAME ───
ast_df_all = pd.DataFrame(all_ast_records)
print(f"\n  Total AST records extracted: {len(ast_df_all)}")
print(f"  Unique patients with AST data: {ast_df_all['patient_id'].nunique()}")
print(f"  Unique organisms: {ast_df_all['organism'].nunique()}")
print(f"  Unique antibiotics: {ast_df_all['antibiotic'].nunique()}")

# Save full AST matrix
ast_df_all.to_csv(os.path.join(DATA_DIR, 'antibiogram_matrix.csv'), index=False)

# ─── COMPUTE PER-PATIENT RESISTOME FEATURES ───
print("\n  Computing per-patient resistome features...")

# Load clinical backbone to get all patient IDs
clinical = pd.read_csv(os.path.join(DATA_DIR, 'clinical_backbone.csv'))
resistome_features = pd.DataFrame({'SR NO.': clinical['SR NO.']})

# Initialize all resistome features to 0 (non-infected patients stay at 0)
resistome_cols = [
    'total_organisms', 'gram_negative_count', 'gram_positive_count',
    'is_polymicrobial', 'has_MDR', 'has_carbapenem_resistance',
    'has_bloodstream_infection', 'has_CSF_infection', 'has_respiratory_infection',
    'has_invasive_infection', 'ARI_score', 'colistin_only_active',
    'gram_negative_dominant', 'n_resistant_classes'
]
for col in resistome_cols:
    resistome_features[col] = 0.0

if len(ast_df_all) > 0:
    for pid in ast_df_all['patient_id'].dropna().unique():
        pid = int(pid)
        pat_data = ast_df_all[ast_df_all['patient_id'] == pid]
        idx = resistome_features[resistome_features['SR NO.'] == pid].index

        if len(idx) == 0:
            continue
        idx = idx[0]

        # Organism counts
        unique_orgs = pat_data['organism'].unique()
        resistome_features.loc[idx, 'total_organisms'] = len(unique_orgs)
        gn_count = sum(1 for o in unique_orgs if GRAM_CLASSIFICATION.get(o) == 'Gram-negative')
        gp_count = sum(1 for o in unique_orgs if GRAM_CLASSIFICATION.get(o) == 'Gram-positive')
        resistome_features.loc[idx, 'gram_negative_count'] = gn_count
        resistome_features.loc[idx, 'gram_positive_count'] = gp_count
        resistome_features.loc[idx, 'is_polymicrobial'] = 1 if len(unique_orgs) >= 2 else 0
        resistome_features.loc[idx, 'gram_negative_dominant'] = 1 if gn_count > gp_count else 0

        # Site-specific infection flags
        sites = pat_data['site'].unique()
        resistome_features.loc[idx, 'has_bloodstream_infection'] = 1 if 'blood' in sites else 0
        resistome_features.loc[idx, 'has_CSF_infection'] = 1 if 'csf' in sites else 0
        resistome_features.loc[idx, 'has_respiratory_infection'] = 1 if ('sputum' in sites or 'tracheal' in sites) else 0
        resistome_features.loc[idx, 'has_invasive_infection'] = 1 if ('blood' in sites or 'csf' in sites) else 0

        # Resistance metrics
        total_tests = len(pat_data)
        r_count = (pat_data['sensitivity'] == 'R').sum()
        s_count = (pat_data['sensitivity'] == 'S').sum()
        ms_count = (pat_data['sensitivity'] == 'MS').sum()

        if total_tests > 0:
            resistome_features.loc[idx, 'ARI_score'] = r_count / total_tests

        # MDR: resistant to agents in >= 3 antibiotic classes
        resistant_classes = pat_data[pat_data['sensitivity'] == 'R']['antibiotic_class'].unique()
        n_resistant_classes = len([c for c in resistant_classes if c != 'Other'])
        resistome_features.loc[idx, 'n_resistant_classes'] = n_resistant_classes
        resistome_features.loc[idx, 'has_MDR'] = 1 if n_resistant_classes >= 3 else 0

        # Carbapenem resistance
        carbapenem_data = pat_data[pat_data['antibiotic_class'] == 'Carbapenems']
        if len(carbapenem_data) > 0:
            carb_r = (carbapenem_data['sensitivity'] == 'R').sum()
            resistome_features.loc[idx, 'has_carbapenem_resistance'] = 1 if carb_r > 0 else 0

        # Colistin-only active (all non-colistin antibiotics are R, but colistin is S/MS)
        non_colistin = pat_data[pat_data['antibiotic'] != 'Colistin']
        colistin = pat_data[pat_data['antibiotic'] == 'Colistin']
        if len(non_colistin) > 0 and len(colistin) > 0:
            all_non_colistin_r = (non_colistin['sensitivity'] == 'R').all()
            colistin_active = colistin['sensitivity'].isin(['S', 'MS']).any()
            resistome_features.loc[idx, 'colistin_only_active'] = 1 if (all_non_colistin_r and colistin_active) else 0

# Save resistome features
resistome_features.to_csv(os.path.join(DATA_DIR, 'resistome_features.csv'), index=False)

# Print summary
print(f"\n  Resistome Feature Summary:")
for col in resistome_cols:
    nonzero = (resistome_features[col] > 0).sum()
    if nonzero > 0:
        print(f"    {col}: {nonzero} patients with non-zero values, mean={resistome_features[col].mean():.3f}")

# Print AST summary statistics
if len(ast_df_all) > 0:
    print(f"\n  Sensitivity Distribution:")
    print(f"    R (Resistant): {(ast_df_all['sensitivity']=='R').sum()}")
    print(f"    S (Sensitive): {(ast_df_all['sensitivity']=='S').sum()}")
    print(f"    MS (Moderately Sensitive): {(ast_df_all['sensitivity']=='MS').sum()}")

    print(f"\n  Organisms (standardized):")
    for org in sorted(ast_df_all['organism'].unique()):
        gram = GRAM_CLASSIFICATION.get(org, '?')
        count = (ast_df_all['organism'] == org).sum()
        print(f"    {org} [{gram}]: {count} AST records")

print(f"\n[Module 2] COMPLETE")
print(f"  Saved antibiogram_matrix.csv ({len(ast_df_all)} records)")
print(f"  Saved resistome_features.csv ({len(resistome_features)} patients)")
