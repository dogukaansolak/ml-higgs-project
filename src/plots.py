import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os

def plot_and_save_roc_curve(outer_roc_data, model_name, save_dir="results"):
    """
    Nested CV dış döngüsünden gelen gerçek etiketleri ve tahmin olasılıklarını
    kullanarak ROC eğrisini çizer ve belirtilen klasöre kaydeder.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    all_y_true = []
    all_y_probs = []
    
    for y_true_fold, y_probs_fold in outer_roc_data:
        all_y_true.extend(y_true_fold)
        all_y_probs.extend(y_probs_fold)
        
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    
    fpr, tpr, _ = roc_curve(all_y_true, all_y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Rastgele tahmin çizgisi
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
    plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
    plt.title(f'{model_name} Modeli - Nested CV ROC Eğrisi')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    file_path = os.path.join(save_dir, f"{model_name.lower()}_roc_curve.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f" Grafiğiniz başarıyla kaydedildi: {file_path}")
    return roc_auc