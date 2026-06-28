import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from feature_selection import apply_feature_selection

def get_hyperparameters():
    """Ödev standartlarına uygun ama optimize edilmiş hafif parametre aralıkları."""
    return {
        'KNN': [{'n_neighbors': k} for k in [5, 9]],
        'SVM': [{'C': c} for c in [0.1, 1]],
        'MLP': [
            {'hidden_layer_sizes': (50,), 'activation': 'relu'},
            {'hidden_layer_sizes': (50,), 'activation': 'tanh'}
        ],
        'XGBoost': [
            {'max_depth': 3, 'learning_rate': 0.1, 'eval_metric': 'logloss'},
            {'max_depth': 5, 'learning_rate': 0.1, 'eval_metric': 'logloss'}
        ]
    }

def initialize_model(model_name, params):
    if model_name == 'KNN':
        return KNeighborsClassifier(**params)
    elif model_name == 'SVM':
        # Standart SVC 100.000 veride kilitlenir. 
        # LinearSVC + CalibratedClassifierCV hem çok hızlıdır hem de ROC için predict_proba üretir.
        base_svm = LinearSVC(**params, dual=False, random_state=42, max_iter=1000)
        return CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)
    elif model_name == 'MLP':
        # Erken durdurma (early_stopping) ekleyerek modelin saatlerce dönmesini engelliyoruz
        return MLPClassifier(**params, max_iter=50, early_stopping=True, random_state=42)
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(**params, random_state=42, n_estimators=50)
    else:
        raise ValueError("Bilinmeyen model!")

def run_nested_cv_for_model(X, y, model_name):
    print(f"\n=== {model_name} Modeli İçin Nested CV Başlatıldı ===")
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    param_grid = get_hyperparameters()[model_name]
    outer_metrics = []
    outer_roc_data = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_arr, y_arr)):
        print(f" > Dış Katlama (Outer Fold) {fold_idx + 1}/5 yürütülüyor...")
        
        X_train_outer, X_test_outer = X_arr[train_idx], X_arr[test_idx]
        y_train_outer, y_test_outer = y_arr[train_idx], y_arr[test_idx]
        
        best_params = None
        best_val_score = -1
        
        for params in param_grid:
            inner_scores = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):
                X_train_inner, X_val_inner = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
                y_train_inner, y_val_inner = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]
                
                scaler = MinMaxScaler()
                X_train_inner_scaled = scaler.fit_transform(X_train_inner)
                X_val_inner_scaled = scaler.transform(X_val_inner)
                
                X_train_inner_sel, X_val_inner_sel, _ = apply_feature_selection(
                    X_train_inner_scaled, y_train_inner, X_val_inner_scaled, k=15
                )
                
                clf = initialize_model(model_name, params)
                clf.fit(X_train_inner_sel, y_train_inner)
                
                preds = clf.predict(X_val_inner_sel)
                inner_scores.append(accuracy_score(y_val_inner, preds))
                
            avg_val_score = np.mean(inner_scores)
            if avg_val_score > best_val_score:
                best_val_score = avg_val_score
                best_params = params
                
        print(f"   -> En İyi Parametre: {best_params} | Başarım: {best_val_score:.4f}")
        
        scaler_outer = MinMaxScaler()
        X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler_outer.transform(X_test_outer)
        
        X_train_outer_sel, X_test_outer_sel, _ = apply_feature_selection(
            X_train_outer_scaled, y_train_outer, X_test_outer_scaled, k=15
        )
        
        final_model = initialize_model(model_name, best_params)
        final_model.fit(X_train_outer_sel, y_train_outer)
        
        test_preds = final_model.predict(X_test_outer_sel)
        test_probs = final_model.predict_proba(X_test_outer_sel)[:, 1]
        
        acc = accuracy_score(y_test_outer, test_preds)
        prec = precision_score(y_test_outer, test_preds, zero_division=0)
        rec = recall_score(y_test_outer, test_preds)
        f1 = f1_score(y_test_outer, test_preds)
        auc = roc_auc_score(y_test_outer, test_probs)
        
        outer_metrics.append([acc, prec, rec, f1, auc])
        outer_roc_data.append((y_test_outer, test_probs))
        
    mean_metrics = np.mean(outer_metrics, axis=0)
    summary_df = pd.DataFrame([mean_metrics], columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])
    summary_df.insert(0, 'Model', model_name)
    
    return summary_df, outer_roc_data