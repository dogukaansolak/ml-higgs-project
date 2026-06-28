import sys
import os
import pandas as pd


sys.path.append(os.path.abspath('src'))

from preprocessing import load_and_sample_data, handle_outliers_iqr
from models import run_nested_cv_for_model
from plots import plot_and_save_roc_curve

def main():
    print("HIGGS ML PIPELINE NİHAİ ÇALIŞMA BAŞLATILIYOR")
   

    data_path = "data/HIGGS.csv"
    if not os.path.exists(data_path):
        print(f"❌ HATA: '{data_path}' konumunda veri seti bulunamadı!")
        return

 
    sample_size = 100000 
    
    print(f"\n[AŞAMA 1] Veri Ön İşleme Başlıyor (Boyut: {sample_size})...")
    df_raw = load_and_sample_data(data_path, sample_size=sample_size)
    df_cleaned = handle_outliers_iqr(df_raw)
    
    X = df_cleaned.iloc[:, 1:].reset_index(drop=True)
    y = df_cleaned.iloc[:, 0].reset_index(drop=True)
    
  
    all_results = []
    

    models_to_run = ['KNN', 'XGBoost', 'MLP', 'SVM']
    
    print("\n[AŞAMA 2] Modelleme ve Nested CV Döngüleri Başlıyor...")
    
    for model_name in models_to_run:
        try:
          
            results_df, roc_data = run_nested_cv_for_model(X, y, model_name)
            all_results.append(results_df)
            
           
            print(f" > {model_name} için ROC Eğrisi çiziliyor...")
            plot_and_save_roc_curve(roc_data, model_name, save_dir="results")
            
        except Exception as e:
            print(f"❌ HATA: {model_name} modeli çalıştırılırken bir sorun oluştu:\n{e}")
            continue
            

  
    print("ÖDEV NİHAİ PERFORMANS MATRİSİ TABLOSU")

    
    if all_results:
        final_summary_df = pd.concat(all_results, ignore_index=True)
        print(final_summary_df.to_string(index=False))
        
        final_summary_df.to_csv("results/final_performance_metrics.csv", index=False)
        print("\n Performans tablosu 'results/final_performance_metrics.csv' olarak kaydedildi.")
    else:
        print("Mevcut hiçbir model başarıyla tamamlanamadı.")
        

    print("TÜM PIPELINE SÜRECİ BAŞARIYLA BİTTİ")


if __name__ == "__main__":
    main()