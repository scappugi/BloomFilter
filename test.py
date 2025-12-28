import os
import time
import multiprocessing
from time import perf_counter

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

    print(f"\n\n[Sequenziale] Iniziato...", end="", flush=True)

    bf = BloomFilter.BloomFilter.from_probability(n, p)
    em = EmailManager.EmailManager()
    avg_times = []
    n_runs = 3
    for _ in range(n_runs):
        start = time.perf_counter()

        for raw_email in dataset:
            email = em.normalize_email(raw_email)
            bf.add(email)

        end = time.perf_counter()
        elapsed = end - start
        avg_times.append(elapsed)
    avg_time = sum(avg_times) / len(avg_times)


    print(f"\n Tempo medio: {avg_time:.4f}s")
    return bf, avg_time


def run_parallel(dataset, n, p, np = multiprocessing.cpu_count()):

    print(f"\n\n[Parallelo]   Iniziato...", end="", flush=True)

    orch = orchestrator.BloomOrchestrator(n, p, np)
    n_runs = 3
    avg_times = []
    bf = BloomFilter

    for i in range(n_runs):
        print (f"\n Esecuzione MapReduce n: {i}", end="", flush=True)
        start = time.time()

        bf = orch.process_chunks(dataset)

        end = time.time()
        elapsed = end - start
        avg_times.append(elapsed)
    avg_time = sum(avg_times) / len(avg_times)


    print(f"\n Tempo medio: {avg_time:.4f}s")
    return bf, avg_time

def run_parallel_shared_memory(dataset, n, p, np = multiprocessing.cpu_count()):

    print(f"\n\n[Parallelo Shared Mem]   Iniziato...", end="", flush=True)

    orch = orchestrator.BloomOrchestrator(n, p, np)
    n_runs = 3
    avg_times = []
    bf = BloomFilter

    for i in range(n_runs):
        print (f"\n Esecuzione Shared Mem {i}", end="", flush=True)
        start = time.time()

        chunk = orch.split_data(dataset)
        bf = orch.run_worker(chunk)


        end = time.time()
        elapsed = end - start
        avg_times.append(elapsed)
    avg_time = sum(avg_times) / len(avg_times)


    print(f"\n Tempo medio: {avg_time:.4f}s")
    return bf, avg_time


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


def compare_performance(bf_seq, bf_par, bf_par_shared, training_dataset, test_size=10000):
    """
    Confronta i TRE filtri usando LO STESSO dataset di test.
    """
    print(f"\n--- CONFRONTO ACCURATEZZA (Test su {test_size} email) ---")

    print("Creazione indice di verità (Set)...")
    dataset_set = set(training_dataset)

    print(f"Generazione dataset di test comune ({test_size} email)...")
    em = EmailManager.EmailManager()
    test_emails = em.generate_complex_email(test_size)

    # 1. Test Sequenziale
    print("Test filtro Sequenziale...", end="")
    tp_seq, fp_seq = evaluate_filter(bf_seq, test_emails, dataset_set, em)
    print(" Fatto.")

    # 2. Test Parallelo Standard
    print("Test filtro Parallelo (Std)...  ", end="")
    tp_par, fp_par = evaluate_filter(bf_par, test_emails, dataset_set, em)
    print(" Fatto.")

    # 3. Test Parallelo Shared Memory
    print("Test filtro Parallelo (Shared)...  ", end="")
    tp_shared, fp_shared = evaluate_filter(bf_par_shared, test_emails, dataset_set, em)
    print(" Fatto.")

    # --- STAMPA TABELLA ---
    header_len = 65
    print("\n" + "=" * header_len)
    print(f"{'METRICA':<20} | {'SEQ':<10} | {'PAR(Std)':<10} | {'PAR(Shm)':<10}")
    print("=" * header_len)

    print(f"{'True Positives':<20} | {tp_seq:<10} | {tp_par:<10} | {tp_shared:<10}")
    print(f"{'False Positives':<20} | {fp_seq:<10} | {fp_par:<10} | {fp_shared:<10}")

    fpr_seq = fp_seq / test_size
    fpr_par = fp_par / test_size
    fpr_shared = fp_shared / test_size

    print(f"{'False Positive Rate':<20} | {fpr_seq:.4f}     | {fpr_par:.4f}     | {fpr_shared:.4f}")

    print("-" * header_len)
    print(f"{'Size (m)':<20} | {bf_seq.get_size():<10} | {bf_par.get_size():<10} | {bf_par_shared.get_size():<10}")
    print(
        f"{'Hash count (k)':<20} | {bf_seq.get_hash_count():<10} | {bf_par.get_hash_count():<10} | {bf_par_shared.get_hash_count():<10}")
    print("=" * header_len)

    # Verifica Identità Matematica su tutti e 3
    if (tp_seq == tp_par == tp_shared) and (fp_seq == fp_par == fp_shared):
        print("\n✅ SUCCESSO: Tutti i tre filtri sono matematicamente IDENTICI.")
    else:
        print("\n⚠️ ATTENZIONE: I risultati differiscono! C'è un bug in una delle implementazioni.")

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

        bf_par, t_par = run_parallel(dataset, N_EMAILS, PROBABILITY,)

        bf_par_shared, t_par_shared = run_parallel_shared_memory(dataset, N_EMAILS, PROBABILITY)

        speedup = t_seq / t_par
        print(f"\n SPEEDUP: {speedup:.2f}x")
        if speedup > 1:
            print(f"   (Il parallelo è {speedup:.2f} volte più veloce)")
        else:
            print("   (Il parallelo è più lento: overhead > guadagno)")

        speedup_shared = t_seq / t_par_shared
        print(f"\n SPEEDUP Shared Mem: {speedup_shared:.2f}x")
        if speedup_shared > 1:
            print(f"   (Il parallelo Shared Mem è {speedup_shared:.2f} volte più veloce)")
        else:
            print("   (Il parallelo Shared Mem è più lento: overhead > guadagno)")

        #compare_performance(bf_seq, bf_par, bf_par_shared, dataset)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

