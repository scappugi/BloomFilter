import os
import time
import multiprocessing
import BloomFilter
import EmailManager
import orchestrator
import csv

def load_dataset_from_csv(filename):
    if not os.path.exists(filename):
        print(f" ERRORE: Il file {filename} non esiste. Esegui prima 'generate_datasets.py'.")
        return None

    print(f"Caricamento {filename} in memoria...", end="", flush=True)
    dataset = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Salta header se presente
        for row in reader:
            if row: # Evita righe vuote
                dataset.append(row[0])
    print(f" Fatto. ({len(dataset)} email caricate)")
    return dataset

def run_sequential(dataset, n, p):

    print(f"[Sequenziale] Iniziato...", end="", flush=True)

    bf = BloomFilter.BloomFilter.from_probability(n, p)
    em = EmailManager.EmailManager()

    start = time.time()

    for raw_email in dataset:
        email = em.normalize_email(raw_email)
        bf.add(email)

    end = time.time()
    elapsed = end - start

    print(f" Fatto in {elapsed:.4f}s")
    return bf, elapsed


def run_parallel(dataset, n, p):

    print(f"[Parallelo]   Iniziato...", end="", flush=True)

    orch = orchestrator.BloomOrchestrator(n, p)

    start = time.time()

    bf = orch.process_chunks(dataset)

    end = time.time()
    elapsed = end - start

    print(f" Fatto in {elapsed:.4f}s")
    return bf, elapsed


def print_stats(bf, nome_algoritmo):
    print(f"\n--- Statistiche {nome_algoritmo} ---")
    print(f"Size (m): {bf.get_size()}")  #
    print(f"Hash functions (k): {bf.get_hash_count()}")
    print(f"False positive rate (p): {bf.get_false_positive_rate()}")


def evaluate_filter(bloom_filter, test_emails, ground_truth_set, em):
    """
    Funzione helper che calcola TP/FP per un singolo filtro
    """
    true_positives = 0
    false_positives = 0

    for raw_email in test_emails:
        email = em.normalize_email(raw_email)

        # Il test che hai richiesto
        if bloom_filter.contains(email):
            if email in ground_truth_set:
                true_positives += 1
            else:
                false_positives += 1
    return true_positives, false_positives
def compare_performance(bf_seq, bf_par, training_dataset, test_size=10000):
    """
    Confronta i due filtri usando LO STESSO dataset di test.
    """
    print(f"\n--- CONFRONTO ACCURATEZZA (Test su {test_size} email) ---")

    print("Creazione indice di verità (Set)...")
    dataset_set = set(training_dataset)

    print(f"Generazione dataset di test comune ({test_size} email)...")
    em = EmailManager.EmailManager()
    test_emails = em.generate_complex_email(test_size)

    print("Test filtro Sequenziale...", end="")
    tp_seq, fp_seq = evaluate_filter(bf_seq, test_emails, dataset_set, em)
    print(" Fatto.")

    print("Test filtro Parallelo...  ", end="")
    tp_par, fp_par = evaluate_filter(bf_par, test_emails, dataset_set, em)
    print(" Fatto.")

    print("\n" + "=" * 45)
    print(f"{'METRICA':<20} | {'SEQUENZIALE':<10} | {'PARALLELO':<10}")
    print("=" * 45)
    print(f"{'True Positives':<20} | {tp_seq:<10} | {tp_par:<10}")
    print(f"{'False Positives':<20} | {fp_seq:<10} | {fp_par:<10}")

    fpr_seq = fp_seq / test_size
    fpr_par = fp_par / test_size
    print(f"{'False Positive Rate':<20} | {fpr_seq:.4f}     | {fpr_par:.4f}")

    print("-" * 45)
    print(f"{'Size (m)':<20} | {bf_seq.get_size():<10} | {bf_par.get_size():<10}")
    print(f"{'Hash count (k)':<20} | {bf_seq.get_hash_count():<10} | {bf_par.get_hash_count():<10}")
    print("=" * 45)

    if tp_seq == tp_par and fp_seq == fp_par:
        print("\nSUCCESSO: I due filtri sono matematicamente IDENTICI.")
    else:
        print("\nATTENZIONE: I risultati differiscono! C'è un bug nella logica parallela.")


def main():
    PROBABILITY = 0.01
    DATASETS_FILES = ["dataset_10k.csv","dataset_100k.csv","dataset_500k.csv", "dataset_1.5m.csv", "dataset_3m.csv", "dataset_5m.csv", "dataset_10m.csv"]

    print(f"--- BENCHMARK AUTOMATICO (CPU Cores: {multiprocessing.cpu_count()}) ---")

    for filename in DATASETS_FILES:
        print("\n" + "=" * 60)
        print(f" DATASET: {filename}")
        print("=" * 60)

        dataset = load_dataset_from_csv(filename)
        if dataset is None: continue  # Salta se file mancante

        N_EMAILS = len(dataset)

        bf_seq, t_seq = run_sequential(dataset, N_EMAILS, PROBABILITY)

        bf_par, t_par = run_parallel(dataset, N_EMAILS, PROBABILITY)

        speedup = t_seq / t_par
        print(f"\n SPEEDUP: {speedup:.2f}x")
        if speedup > 1:
            print(f"   (Il parallelo è {speedup:.2f} volte più veloce)")
        else:
            print("   (Il parallelo è più lento: overhead > guadagno)")

      #  compare_performance(bf_seq, bf_par, dataset)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

