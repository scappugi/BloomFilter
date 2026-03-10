import os
import sys
import csv
import matplotlib.pyplot as plt

# --- Blocco di Correzione Percorsi ---
# Aggiunge la root del progetto al sys.path per rendere gli import robusti
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fine Blocco ---

from src import EmailManager

# Definiamo la directory dei dati in modo robusto
DATA_DIR = os.path.join(project_root, "data")

def load_dataset_from_csv(filename):
    """
    Carica un dataset da un file CSV situato nella directory DATA_DIR.
    """
    full_path = os.path.join(DATA_DIR, os.path.basename(filename))

    if not os.path.exists(full_path):
        print(f" ERRORE: Il file {full_path} non esiste. Esegui prima 'scripts/generate_datasets.py'.")
        return None

    print(f"Caricamento {full_path} in memoria...", end="", flush=True)
    dataset = []
    with open(full_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Salta header
        for row in reader:
            if row:
                dataset.append(row[0])
    print(f" Fatto. ({len(dataset)} email caricate)")
    return dataset

def evaluate_filter(bloom_filter, test_emails, ground_truth_set):
    """
    Calcola TP/FP per un singolo filtro, normalizzando le email.
    """
    true_positives = 0
    false_positives = 0
    em = EmailManager.EmailManager()

    for raw_email in test_emails:
        email = em.normalize_email(raw_email)
        if bloom_filter.contains(email):
            if email in ground_truth_set:
                true_positives += 1
            else:
                false_positives += 1
    return true_positives, false_positives

def compare_performance(filters_dict, training_dataset, test_size=10000):
    """
    Funzione generica per confrontare l'accuratezza di un dizionario di filtri.
    filters_dict: {'Nome Etichetta': bloom_filter_object}
    """
    print(f"\n--- CONFRONTO ACCURATEZZA (Test su {test_size} email) ---")
    
    dataset_set = set(training_dataset)
    em = EmailManager.EmailManager()
    test_emails = em.generate_complex_email(test_size)

    results = {}
    for label, bf in filters_dict.items():
        print(f"Test filtro {label}... ", end="", flush=True)
        tp, fp = evaluate_filter(bf, test_emails, dataset_set)
        results[label] = {'tp': tp, 'fp': fp, 'bf': bf}
        print("Fatto.")

    # Stampa Tabella
    labels = list(results.keys())
    header = f"{'METRICA':<20} | " + " | ".join([f"{l:<10}" for l in labels])
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    tp_line = f"{'True Positives':<20} | " + " | ".join([f"{results[l]['tp']:<10}" for l in labels])
    fp_line = f"{'False Positives':<20} | " + " | ".join([f"{results[l]['fp']:<10}" for l in labels])
    print(tp_line)
    print(fp_line)

    fpr_values = [(results[l]['fp'] / test_size) for l in labels]
    fpr_line = f"{'False Positive Rate':<20} | " + " | ".join([f"{fpr:<10.4f}" for fpr in fpr_values])
    print(fpr_line)
    
    print("-" * len(header))
    size_line = f"{'Size (m)':<20} | " + " | ".join([f"{results[l]['bf'].get_size():<10}" for l in labels])
    print(size_line)
    print("=" * len(header))

    # Verifica identità
    all_tp_equal = all(res['tp'] == results[labels[0]]['tp'] for res in results.values())
    all_fp_equal = all(res['fp'] == results[labels[0]]['fp'] for res in results.values())

    if all_tp_equal and all_fp_equal:
        print("\n✅ SUCCESSO: Tutti i filtri sono matematicamente IDENTICI.")
    else:
        print("\n⚠️ ATTENZIONE: I risultati differiscono!")

def plot_scalability(dataset_name, worker_counts, seq_time, results_dict):
    """
    Funzione generica per plottare tempi e speedup.
    results_dict: {'Nome Etichetta': [lista_tempi]}
    """
    plt.figure(figsize=(12, 5))

    # --- Grafico 1: Tempi Assoluti ---
    plt.subplot(1, 2, 1)
    plt.axhline(y=seq_time, color='r', linestyle='-', label=f"Sequenziale ({seq_time:.2f}s)")
    
    markers = ['o', 's', '^', '*', 'D', 'P']
    for i, (label, times) in enumerate(results_dict.items()):
        plt.plot(worker_counts, times, marker=markers[i % len(markers)], label=label)

    plt.title(f"Tempi di Esecuzione: {dataset_name}")
    plt.xlabel("Numero di Worker/Thread")
    plt.ylabel("Tempo (s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # --- Grafico 2: Speedup ---
    plt.subplot(1, 2, 2)
    
    for i, (label, times) in enumerate(results_dict.items()):
        speedup = [seq_time / t for t in times]
        plt.plot(worker_counts, speedup, marker=markers[i % len(markers)], label=label)
    
    plt.plot(worker_counts, worker_counts, 'k--', alpha=0.3, label="Ideale")
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)

    plt.title(f"Speedup: {dataset_name}")
    plt.xlabel("Numero di Worker/Thread")
    plt.ylabel("Speedup (Nx)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()
