import pandas as pd
import numpy as np

def load_and_sample_data(filepath, sample_size=100000):
    """
    HIGGS veri setini okur ve analiz için rastgele bir alt küme seçer.
    HIGGS veri setinde ilk sütun hedef değişkendir (0 veya 1).
    """
    print(f" Veri seti yükleniyor: {filepath}")
    
    df_chunk = pd.read_csv(filepath, header=None, nrows=1000000) 
    
    print(f" Toplam okunan veri içinden {sample_size} adet rastgele örnek seçiliyor...")
    sampled_df = df_chunk.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return sampled_df

def handle_outliers_iqr(df):
    """
    IQR yöntemi ile aykırı değerleri tespit eder ve sınır değerlerle değiştirir.
    """
    print(" IQR yöntemi ile aykırı değer analizi ve baskılama başlatıldı...")
    df_cleaned = df.copy()
    
    feature_cols = df_cleaned.columns[1:]
    outlier_count = 0
    
    for col in feature_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        outlier_count += len(outliers)
        
        df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
        
    print(f" Aykırı değer analizi tamamlandı. Toplam {outlier_count} hücre sınır değerlere çekildi.")
    return df_cleaned