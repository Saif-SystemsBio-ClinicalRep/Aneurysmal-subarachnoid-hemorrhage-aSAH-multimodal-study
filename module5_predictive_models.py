"""
Module 5: Predictive Modeling & Stacked Ensemble
==================================================
Trains 7 ML models via Repeated Nested Stratified 5-Fold CV,
builds a Stacked Ensemble Meta-Learner, and computes all evaluation
metrics with bootstrap 95% CIs.
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             f1_score, recall_score, precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

PROJECT_DIR = r"C:\Users\SAIF\.gemini\antigravity-ide\scratch\sah_aneurysm_study"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "output", "models")
TABLE_DIR = os.path.join(PROJECT_DIR, "output", "tables")

print("[Module 5] Loading feature matrix...")
unified = pd.read_csv(os.path.join(DATA_DIR, 'unified_feature_matrix.csv'))
feature_names = open(os.path.join(DATA_DIR, 'feature_names.txt')).read().strip().split('\n')

X = unified[feature_names].values
y = unified['outcome_binary'].values
feature_names_arr = np.array(feature_names)

print(f"  X shape: {X.shape}, y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ─── DEFINE MODELS ───
scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
print(f"  Class imbalance ratio: {scale_pos_weight:.2f}")

def get_models():
    """Return dictionary of model name -> model instance."""
    models = {
        'Firth_LR': LogisticRegression(
            penalty='l2', C=1.0, class_weight='balanced',
            max_iter=2000, solver='lbfgs', random_state=42
        ),
        'ElasticNet_LR': LogisticRegression(
            penalty='elasticnet', l1_ratio=0.5, C=0.5,
            class_weight='balanced', max_iter=2000,
            solver='saga', random_state=42
        ),
        'SVM_RBF': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True,
            random_state=42
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, num_leaves=8,
            learning_rate=0.05, is_unbalance=True,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0,
            random_state=42, n_jobs=-1, verbose=-1
        ),
    }
    return models

# ─── REPEATED NESTED STRATIFIED K-FOLD CV ───
print("\n  Running Repeated Nested 5-Fold CV (10 repetitions = 50 evaluations)...")

N_OUTER_FOLDS = 5
N_REPEATS = 10
N_BOOTSTRAP = 500

outer_cv = RepeatedStratifiedKFold(n_splits=N_OUTER_FOLDS, n_repeats=N_REPEATS, random_state=42)

# Store results
all_results = {name: {
    'roc_auc': [], 'pr_auc': [], 'brier': [], 'sensitivity': [],
    'specificity': [], 'f1': [], 'ppv': [], 'npv': [],
    'y_true_all': [], 'y_prob_all': []
} for name in get_models().keys()}

# Add stacked ensemble
all_results['Stacked_Ensemble'] = {
    'roc_auc': [], 'pr_auc': [], 'brier': [], 'sensitivity': [],
    'specificity': [], 'f1': [], 'ppv': [], 'npv': [],
    'y_true_all': [], 'y_prob_all': []
}

fold_count = 0
for train_idx, test_idx in outer_cv.split(X, y):
    fold_count += 1
    if fold_count % 10 == 0:
        print(f"    Fold {fold_count}/{N_OUTER_FOLDS * N_REPEATS}...")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Store base model predictions for stacking
    base_train_preds = {}
    base_test_preds = {}
    
    # Train and evaluate each base model
    models = get_models()
    for name, model in models.items():
        try:
            # Models that need scaled data
            if name in ['Firth_LR', 'ElasticNet_LR', 'SVM_RBF']:
                model.fit(X_train_s, y_train)
                y_prob = model.predict_proba(X_test_s)[:, 1]
                # For stacking: out-of-fold predictions on training set
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                oof_preds = cross_val_predict(model.__class__(**model.get_params()),
                                              X_train_s, y_train, cv=inner_cv,
                                              method='predict_proba')[:, 1]
                base_train_preds[name] = oof_preds
                base_test_preds[name] = y_prob
            else:
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                oof_preds = cross_val_predict(model.__class__(**model.get_params()),
                                              X_train, y_train, cv=inner_cv,
                                              method='predict_proba')[:, 1]
                base_train_preds[name] = oof_preds
                base_test_preds[name] = y_prob
            
            y_pred = (y_prob >= 0.5).astype(int)
            
            # Compute metrics
            if len(np.unique(y_test)) > 1:
                roc = roc_auc_score(y_test, y_prob)
                pr = average_precision_score(y_test, y_prob)
            else:
                roc = np.nan
                pr = np.nan
            brier = brier_score_loss(y_test, y_prob)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
            npv = tn / max(tn + fn, 1) if (tn + fn) > 0 else 0
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            all_results[name]['roc_auc'].append(roc)
            all_results[name]['pr_auc'].append(pr)
            all_results[name]['brier'].append(brier)
            all_results[name]['sensitivity'].append(sens)
            all_results[name]['specificity'].append(spec)
            all_results[name]['f1'].append(f1)
            all_results[name]['ppv'].append(ppv)
            all_results[name]['npv'].append(npv)
            all_results[name]['y_true_all'].extend(y_test.tolist())
            all_results[name]['y_prob_all'].extend(y_prob.tolist())
            
        except Exception as e:
            print(f"      WARNING: {name} failed on fold {fold_count}: {e}")
    
    # ─── STACKED ENSEMBLE META-LEARNER ───
    try:
        if len(base_train_preds) >= 3:
            # Build meta-features
            meta_X_train = np.column_stack([base_train_preds[n] for n in sorted(base_train_preds.keys())])
            meta_X_test = np.column_stack([base_test_preds[n] for n in sorted(base_test_preds.keys())])
            
            # Train meta-learner (penalized logistic regression)
            meta_model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
            meta_model.fit(meta_X_train, y_train)
            
            y_prob_stack = meta_model.predict_proba(meta_X_test)[:, 1]
            y_pred_stack = (y_prob_stack >= 0.5).astype(int)
            
            if len(np.unique(y_test)) > 1:
                roc_s = roc_auc_score(y_test, y_prob_stack)
                pr_s = average_precision_score(y_test, y_prob_stack)
            else:
                roc_s = np.nan
                pr_s = np.nan
            brier_s = brier_score_loss(y_test, y_prob_stack)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_stack, labels=[0,1]).ravel()
            sens_s = tp / max(tp + fn, 1)
            spec_s = tn / max(tn + fp, 1)
            ppv_s = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
            npv_s = tn / max(tn + fn, 1) if (tn + fn) > 0 else 0
            f1_s = f1_score(y_test, y_pred_stack, zero_division=0)
            
            all_results['Stacked_Ensemble']['roc_auc'].append(roc_s)
            all_results['Stacked_Ensemble']['pr_auc'].append(pr_s)
            all_results['Stacked_Ensemble']['brier'].append(brier_s)
            all_results['Stacked_Ensemble']['sensitivity'].append(sens_s)
            all_results['Stacked_Ensemble']['specificity'].append(spec_s)
            all_results['Stacked_Ensemble']['f1'].append(f1_s)
            all_results['Stacked_Ensemble']['ppv'].append(ppv_s)
            all_results['Stacked_Ensemble']['npv'].append(npv_s)
            all_results['Stacked_Ensemble']['y_true_all'].extend(y_test.tolist())
            all_results['Stacked_Ensemble']['y_prob_all'].extend(y_prob_stack.tolist())
    except Exception as e:
        print(f"      WARNING: Stacked Ensemble failed on fold {fold_count}: {e}")

# ─── COMPUTE SUMMARY METRICS WITH BOOTSTRAP CI ───
print("\n  Computing summary metrics with bootstrap 95% CIs...")

def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """Compute bootstrap 95% CI for a metric."""
    values = [v for v in values if not np.isnan(v)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    return np.mean(values), lower, upper

summary_rows = []
for model_name in all_results.keys():
    metrics = all_results[model_name]
    row = {'Model': model_name}
    
    for metric_name in ['roc_auc', 'pr_auc', 'brier', 'sensitivity', 'specificity', 'f1', 'ppv', 'npv']:
        mean_val, lower, upper = bootstrap_ci(metrics[metric_name])
        row[f'{metric_name}_mean'] = mean_val
        row[f'{metric_name}_ci_lower'] = lower
        row[f'{metric_name}_ci_upper'] = upper
        row[f'{metric_name}_formatted'] = f"{mean_val:.3f} [{lower:.3f}-{upper:.3f}]"
    
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Sort by PR-AUC (most important for imbalanced data)
summary_df = summary_df.sort_values('pr_auc_mean', ascending=False)

# Save summary
summary_df.to_csv(os.path.join(TABLE_DIR, 'table3_model_comparison.csv'), index=False)

# Print formatted results
print("\n  MODEL PERFORMANCE COMPARISON (Sorted by PR-AUC)")
print("  " + "=" * 120)
print(f"  {'Model':<22} {'ROC-AUC':<25} {'PR-AUC':<25} {'Brier':<25} {'Sensitivity':<25} {'F1':<25}")
print("  " + "-" * 120)
for _, row in summary_df.iterrows():
    print(f"  {row['Model']:<22} {row['roc_auc_formatted']:<25} {row['pr_auc_formatted']:<25} "
          f"{row['brier_formatted']:<25} {row['sensitivity_formatted']:<25} {row['f1_formatted']:<25}")

# ─── TRAIN FINAL MODELS ON FULL DATA (for SHAP/XAI in Module 6) ───
print("\n  Training final models on full dataset (for XAI)...")

scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)

final_models = {}
models = get_models()
for name, model in models.items():
    try:
        if name in ['Firth_LR', 'ElasticNet_LR', 'SVM_RBF']:
            model.fit(X_full_scaled, y)
        else:
            model.fit(X, y)
        final_models[name] = model
        print(f"    {name}: trained successfully")
    except Exception as e:
        print(f"    {name}: FAILED - {e}")

# Train final stacked ensemble
print("  Training final Stacked Ensemble...")
inner_cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_features_full = np.zeros((len(X), len(final_models)))

for i, (name, model) in enumerate(sorted(final_models.items())):
    try:
        if name in ['Firth_LR', 'ElasticNet_LR', 'SVM_RBF']:
            oof = cross_val_predict(model.__class__(**model.get_params()),
                                    X_full_scaled, y, cv=inner_cv_final,
                                    method='predict_proba')[:, 1]
        else:
            oof = cross_val_predict(model.__class__(**model.get_params()),
                                    X, y, cv=inner_cv_final,
                                    method='predict_proba')[:, 1]
        meta_features_full[:, i] = oof
    except Exception as e:
        print(f"    WARNING: OOF for {name} failed: {e}")

meta_model_final = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
meta_model_final.fit(meta_features_full, y)
final_models['Stacked_Ensemble'] = meta_model_final

# Save predictions and model data
predictions_df = pd.DataFrame({
    'SR_NO': unified['SR NO.'],
    'outcome_true': y,
})

for name, model in final_models.items():
    try:
        if name == 'Stacked_Ensemble':
            predictions_df[f'prob_{name}'] = meta_model_final.predict_proba(meta_features_full)[:, 1]
        elif name in ['Firth_LR', 'ElasticNet_LR', 'SVM_RBF']:
            predictions_df[f'prob_{name}'] = model.predict_proba(X_full_scaled)[:, 1]
        else:
            predictions_df[f'prob_{name}'] = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"    WARNING: Predictions for {name} failed: {e}")

predictions_df.to_csv(os.path.join(DATA_DIR, 'model_predictions.csv'), index=False)

# Save CV results for plotting
cv_plot_data = {}
for model_name in all_results.keys():
    cv_plot_data[model_name] = {
        'y_true': all_results[model_name]['y_true_all'],
        'y_prob': all_results[model_name]['y_prob_all'],
    }

# Save as numpy arrays for figure generation
np.savez(os.path.join(MODEL_DIR, 'cv_results.npz'),
         **{f"{k}_y_true": np.array(v['y_true']) for k, v in cv_plot_data.items()},
         **{f"{k}_y_prob": np.array(v['y_prob']) for k, v in cv_plot_data.items()})

# Save feature importances from tree-based models
print("\n  Extracting feature importances...")
importance_data = {}

# XGBoost feature importance
if 'XGBoost' in final_models:
    xgb_imp = final_models['XGBoost'].feature_importances_
    importance_data['XGBoost'] = dict(zip(feature_names, xgb_imp.tolist()))
    top_xgb = sorted(zip(feature_names, xgb_imp), key=lambda x: x[1], reverse=True)[:10]
    print("    XGBoost Top 10:")
    for f, imp in top_xgb:
        print(f"      {f}: {imp:.4f}")

# LightGBM feature importance
if 'LightGBM' in final_models:
    lgb_imp = final_models['LightGBM'].feature_importances_
    importance_data['LightGBM'] = dict(zip(feature_names, lgb_imp.tolist()))

# Random Forest feature importance
if 'Random_Forest' in final_models:
    rf_imp = final_models['Random_Forest'].feature_importances_
    importance_data['Random_Forest'] = dict(zip(feature_names, rf_imp.tolist()))

# Logistic Regression coefficients
if 'Firth_LR' in final_models:
    lr_coefs = np.abs(final_models['Firth_LR'].coef_[0])
    importance_data['Firth_LR'] = dict(zip(feature_names, lr_coefs.tolist()))

with open(os.path.join(MODEL_DIR, 'feature_importances.json'), 'w') as f:
    json.dump(importance_data, f, indent=2)

# Save scaler for later use
import pickle
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_full, f)
with open(os.path.join(MODEL_DIR, 'final_models.pkl'), 'wb') as f:
    pickle.dump(final_models, f)

print(f"\n[Module 5] COMPLETE")
print(f"  Saved table3_model_comparison.csv")
print(f"  Saved model_predictions.csv")
print(f"  Saved cv_results.npz")
print(f"  Saved feature_importances.json")
print(f"  Saved final_models.pkl")
