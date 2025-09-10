
\"\"\"predict_liver_cirrhosis.py
Comprehensive script to train and evaluate ML models for predicting liver cirrhosis.

Features:
- Loads dataset from CSV (path provided by user) or generates a synthetic dataset if file not found.
- Preprocesses data (imputation, encoding, scaling).
- Trains multiple models: Logistic Regression, Random Forest, SVM, XGBoost.
- Evaluates models with Accuracy, Precision, Recall, F1, ROC-AUC and shows a ROC curve.
- Saves best model to disk using joblib.
- Contains helper functions for reuse in notebooks or pipelines.
- Usage: python predict_liver_cirrhosis.py --data path/to/dataset.csv
\"\"\"

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt

# Try importing XGBoost; if not installed, warn but continue (RandomForest will still work)
try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    XGBClassifier = None
    has_xgb = False

RANDOM_STATE = 42

def generate_synthetic_dataset(n_samples=1000, random_state=RANDOM_STATE):
    \"\"\"Generate a synthetic dataset similar to typical liver datasets for demonstration.\"\"\"
    np.random.seed(random_state)
    age = np.random.randint(20, 80, size=n_samples)
    bilirubin = np.random.lognormal(mean=0.5, sigma=0.8, size=n_samples)  # continuous
    albumin = np.round(np.random.normal(3.5, 0.6, size=n_samples), 2)
    platelets = np.random.randint(50, 400, size=n_samples)
    # Create a binary target with some dependence on features
    risk_score = 0.02*(age-40) + 0.6*(bilirubin>1.2).astype(int) - 0.5*(albumin>3.5).astype(int) + 0.002*(400-platelets)
    prob = 1/(1+np.exp(-risk_score))
    target = (np.random.rand(n_samples) < prob).astype(int)
    df = pd.DataFrame({
        'Age': age,
        'Bilirubin': np.round(bilirubin, 3),
        'Albumin': albumin,
        'Platelets': platelets,
        'Sex': np.random.choice(['M', 'F'], size=n_samples),
        'Target': target
    })
    return df

def load_data(csv_path=None):
    \"\"\"Load dataset from CSV. If not found, fall back to synthetic dataset and warn the user.\"\"\"
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f\"Loaded dataset from {csv_path} with shape {df.shape}\")
    else:
        print(\"Warning: dataset CSV not found. Generating a synthetic demo dataset.\")
        df = generate_synthetic_dataset()
    return df

def preprocess(df, target_col='Target'):
    \"\"\"Preprocess dataframe: handle missing values, encode categorical, scale numeric.\"\"\"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return X, y, preprocessor, numeric_cols, categorical_cols

def train_and_evaluate(X, y, preprocessor, test_size=0.2, random_state=RANDOM_STATE, do_grid_search=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    models = {}
    results = {}

    # Logistic Regression pipeline
    pipe_lr = Pipeline(steps=[('preproc', preprocessor),
                              ('clf', LogisticRegression(max_iter=200, random_state=random_state))])
    pipe_lr.fit(X_train, y_train)
    models['LogisticRegression'] = pipe_lr
    results['LogisticRegression'] = evaluate_model(pipe_lr, X_test, y_test)

    # Random Forest pipeline
    pipe_rf = Pipeline(steps=[('preproc', preprocessor),
                              ('clf', RandomForestClassifier(n_estimators=200, random_state=random_state))])
    pipe_rf.fit(X_train, y_train)
    models['RandomForest'] = pipe_rf
    results['RandomForest'] = evaluate_model(pipe_rf, X_test, y_test)

    # SVM pipeline (probability True for ROC/AUC)
    pipe_svc = Pipeline(steps=[('preproc', preprocessor),
                               ('clf', SVC(probability=True, random_state=random_state))])
    pipe_svc.fit(X_train, y_train)
    models['SVM'] = pipe_svc
    results['SVM'] = evaluate_model(pipe_svc, X_test, y_test)

    # XGBoost if available
    if has_xgb:
        pipe_xgb = Pipeline(steps=[('preproc', preprocessor),
                                   ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state))])
        pipe_xgb.fit(X_train, y_train)
        models['XGBoost'] = pipe_xgb
        results['XGBoost'] = evaluate_model(pipe_xgb, X_test, y_test)
    else:
        print('XGBoost not installed. Skipping XGBoost model.')

    # Choose best model by AUC
    best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'] if results[k]['roc_auc'] is not None else -1)
    best_model_info = results[best_model_name]
    print(f\"\\nBest model: {best_model_name} | AUC: {best_model_info['roc_auc']:.4f}\")

    return models, results, best_model_name, X_test, y_test

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        # some models may not support predict_proba
        try:
            y_proba = pipeline.decision_function(X_test)
        except Exception:
            y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f\"Model evaluation:\\n Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {roc_auc}\")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc, 'y_pred': y_pred, 'y_proba': y_proba}

def plot_roc_curves(results, X_test, y_test, filename='roc_curve.png'):
    plt.figure()
    for name, info in results.items():
        y_proba = info.get('y_proba')
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f\"{name} (AUC={info['roc_auc']:.3f})\")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename)
    print(f\"Saved ROC curve to {filename}\")

def save_best_model(models, best_model_name, out_path='best_model.joblib'):
    best_model = models[best_model_name]
    joblib.dump(best_model, out_path)
    print(f\"Saved best model ({best_model_name}) to {out_path}\")

def main(args):
    df = load_data(args.data)
    # attempt to detect common target column names if user didn't use 'Target'
    possible_targets = ['Target', 'Cirrhosis', 'CirrhosisLabel', 'label', 'target']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        # If there's a column named 'Diagnosis' or similar, try to map it, else assume last col is target
        target_col = df.columns[-1]
        print(f\"Assuming target column is '{target_col}' (last column in dataset). Please verify.\")

    X, y, preprocessor, num_cols, cat_cols = preprocess(df, target_col=target_col)
    models, results, best_model_name, X_test, y_test = train_and_evaluate(X, y, preprocessor)

    # attach y_proba to results for plotting if missing
    for name, mdl in models.items():
        try:
            y_proba = mdl.predict_proba(X_test)[:,1]
            results[name]['y_proba'] = y_proba
        except Exception:
            try:
                results[name]['y_proba'] = mdl.decision_function(X_test)
            except Exception:
                results[name]['y_proba'] = None

    plot_roc_curves(results, X_test, y_test, filename='roc_curve.png')
    save_best_model(models, best_model_name, out_path='best_model.joblib')

    # Save a brief CSV of model metrics
    metrics_df = pd.DataFrame.from_dict({k: {k2: v2 for k2, v2 in res.items() if k2 in ['accuracy','precision','recall','f1','roc_auc']} for k,res in results.items()}).T
    metrics_df.to_csv('model_metrics_summary.csv', index=True)
    print('Saved model_metrics_summary.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML models for liver cirrhosis prediction.')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV dataset. If omitted, a synthetic dataset will be used.')
    args = parser.parse_args()
    main(args)
