from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

def apply_feature_selection(X_train, y_train, X_val, k=15):
    """
    ANOVA F-score yöntemi kullanarak en iyi k (15) adet özelliği seçer.
    Veri sızıntısını önlemek için sadece Train seti üzerinde 'fit' edilir,
    Validation/Test setleri ise sadece 'transform' edilir.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    X_val_selected = selector.transform(X_val)
    
    selected_features_mask = selector.get_support()
    

    if isinstance(X_train, pd.DataFrame):
        selected_names = X_train.columns[selected_features_mask].tolist()
    else:
        selected_names = [f"Feature_{i}" for i in range(X_train.shape[1]) if selected_features_mask[i]]
        
    return X_train_selected, X_val_selected, selected_names