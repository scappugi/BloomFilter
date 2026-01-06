import os
import time
import multiprocessing
from time import perf_counter

import BloomFilter
import EmailManager
import orchestrator
import csv
import plot_utils

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

def run_parallel_joblib(dataset, n, p, np = multiprocessing.cpu_count()):

    print(f"\n\n[Parallelo Joblib]   Iniziato...", end="", flush=True)

    orch = orchestrator.BloomOrchestrator(n, p, np)
    n_runs = 3
    avg_times = []
    bf = BloomFilter

    for i in range(n_runs):
        print (f"\n Esecuzione Joblib n: {i}", end="", flush=True)
        start = time.time()

        bf = orch.run_joblib_worker(dataset)

        end = time.time()
        elapsed = end - start
        avg_times.append(elapsed)
    avg_time = sum(avg_times) / len(avg_times)

    print(f"\n Tempo medio: {avg_time:.4f}s")
    return bf, avg_time

def run_parallel_joblib_shared(dataset, n, p, np = multiprocessing.cpu_count()):

    print(f"\n\n[Parallelo Joblib Shared Mem]   Iniziato...", end="", flush=True)

    orch = orchestrator.BloomOrchestrator(n, p, np)
    n_runs = 3
    avg_times = []
    bf = BloomFilter

    for i in range(n_runs):
        print (f"\n Esecuzione Joblib Shared Mem n: {i}", end="", flush=True)
        start = time.time()

        bf = orch.run_joblib_shared_worker(dataset)

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


def compare_performance(bf_seq, bf_par, bf_par_shared, bf_job, bf_job_sh, training_dataset, test_size=10000):
    """
    Confronta i CINQUE filtri usando LO STESSO dataset di test.

    """
    print(f"\n--- CONFRONTO ACCURATEZZA (Test su {test_size} email) ---")

    print("Creazione indice di verità (Set)...")
    dataset_set = set(training_dataset)

    print(f"Generazione dataset di test comune ({test_size} email)...")
    em = EmailManager.EmailManager()
    test_emails = em.generate_complex_email(test_size)

    # 1. Test Sequenziale
    print("Test filtro Sequenziale...       ", end="")
    tp_seq, fp_seq = evaluate_filter(bf_seq, test_emails, dataset_set, em)
    print(" Fatto.")

    # 2. Test Parallelo Standard (MP)
    print("Test filtro MP (Std)...          ", end="")
    tp_par, fp_par = evaluate_filter(bf_par, test_emails, dataset_set, em)
    print(" Fatto.")

    # 3. Test Parallelo Shared Memory (MP - Vecchio metodo)
    print("Test filtro MP (Shared)...       ", end="")
    tp_shared, fp_shared = evaluate_filter(bf_par_shared, test_emails, dataset_set, em)
    print(" Fatto.")

    # 4. Test Joblib Standard
    print("Test filtro Joblib (Std)...      ", end="")
    tp_job, fp_job = evaluate_filter(bf_job, test_emails, dataset_set, em)
    print(" Fatto.")

    # 5. Test Joblib Shared (NumPy Fast)
    print("Test filtro Joblib (NumPy)...    ", end="")
    tp_job_sh, fp_job_sh = evaluate_filter(bf_job_sh, test_emails, dataset_set, em)
    print(" Fatto.")

    # --- STAMPA TABELLA ---
    # Larghezza adatta a 5 colonne
    header_len = 100
    print("\n" + "=" * header_len)
    # Intestazioni colonne
    print(f"{'METRICA':<20} | {'SEQ':<9} | {'MP(Std)':<9} | {'MP(Shm)':<9} | {'JOB(Std)':<9} | {'JOB(Nmp)':<9}")
    print("=" * header_len)

    # True Positives
    print(f"{'True Positives':<20} | {tp_seq:<9} | {tp_par:<9} | {tp_shared:<9} | {tp_job:<9} | {tp_job_sh:<9}")

    # False Positives
    print(f"{'False Positives':<20} | {fp_seq:<9} | {fp_par:<9} | {fp_shared:<9} | {fp_job:<9} | {fp_job_sh:<9}")

    # False Positive Rate
    fpr_seq = fp_seq / test_size
    fpr_par = fp_par / test_size
    fpr_shared = fp_shared / test_size
    fpr_job = fp_job / test_size
    fpr_job_sh = fp_job_sh / test_size

    print(
        f"{'False Positive Rate':<20} | {fpr_seq:.4f}    | {fpr_par:.4f}    | {fpr_shared:.4f}    | {fpr_job:.4f}    | {fpr_job_sh:.4f}")

    print("-" * header_len)

    # Parametri interni (Size)
    print(
        f"{'Size (m)':<20} | {bf_seq.get_size():<9} | {bf_par.get_size():<9} | {bf_par_shared.get_size():<9} | {bf_job.get_size():<9} | {bf_job_sh.get_size():<9}")

    # Parametri interni (Hash count)
    print(
        f"{'Hash count (k)':<20} | {bf_seq.get_hash_count():<9} | {bf_par.get_hash_count():<9} | {bf_par_shared.get_hash_count():<9} | {bf_job.get_hash_count():<9} | {bf_job_sh.get_hash_count():<9}")

    print("=" * header_len)

    # Verifica Identità Matematica su tutti e 5
    all_tp_equal = (tp_seq == tp_par == tp_shared == tp_job == tp_job_sh)
    all_fp_equal = (fp_seq == fp_par == fp_shared == fp_job == fp_job_sh)

    if all_tp_equal and all_fp_equal:
        print("\n✅ SUCCESSO: Tutti i CINQUE filtri sono matematicamente IDENTICI.")
    else:
        print("\n⚠️ ATTENZIONE: I risultati differiscono! C'è un bug in una delle implementazioni.")
        # Debug helper
        if tp_seq != tp_job_sh:
            print(f"   -> Joblib (NumPy) differisce dal Sequenziale (TP: {tp_job_sh} vs {tp_seq})")

def main():
    PROBABILITY = 0.01
    DATASETS_FILES = ["dataset_10k.csv","dataset_100k.csv","dataset_500k.csv", "dataset_1.5m.csv", "dataset_3m.csv", "dataset_5m.csv", "dataset_10m.csv"]

    print(f"--- BENCHMARK AUTOMATICO (CPU Cores: {multiprocessing.cpu_count()}) ---")

    dt = ["dataset_500k.csv", ]
    dt1 = load_dataset_from_csv("dataset_500k.csv")
    bf_seq, t_seq = run_sequential(dt1, len(dt1), PROBABILITY)
    bf_par, t_par = run_parallel(dt1, len(dt1), PROBABILITY, 8)
    print(f"\n il tempo parallelo è: {t_par} e il tempo sequenziale è: {t_seq}, lo speedup è: {t_seq / t_par}")

    for filename in DATASETS_FILES:
        if filename == "dataset_5m.csv":
            print("\n" + "=" * 60)
            print(f" DATASET: {filename}")
            print("=" * 60)

            dataset = load_dataset_from_csv(filename)
            if dataset is None: continue  # Salta se file mancante

            N_EMAILS = len(dataset)
            t_parallel_times = []
            t_parallel_times_shared = []
            t_parallel_times_joblib = []
            t_parallel_times_joblib_shared = []

            bf_seq, t_seq = run_sequential(dataset, N_EMAILS, PROBABILITY)
            for n_process in range (1, (multiprocessing.cpu_count()*2)+1):
                if n_process >= 14:

                    print(f"\n--- Esecuzione con {n_process} processi ---")

                    bf_par, t_par = run_parallel(dataset, N_EMAILS, PROBABILITY, n_process)
                    t_parallel_times.append(t_par)

                    bf_par_shared, t_par_shared = run_parallel_shared_memory(dataset, N_EMAILS, PROBABILITY, n_process)
                    t_parallel_times_shared.append(t_par_shared)

                    bf_joblib, t_par_joblib = run_parallel_joblib(dataset, N_EMAILS, PROBABILITY, n_process)
                    t_parallel_times_joblib.append(t_par_joblib)

                    bf_joblib_shared, t_par_joblib_shared = run_parallel_joblib_shared(dataset, N_EMAILS, PROBABILITY, n_process)
                    t_parallel_times_joblib_shared.append(t_par_joblib_shared)

            print(f"\nGenerazione grafico scalabilità per {filename}...")

            plot_utils.plot_scalability(
                filename,
                [i for i in range (1,16+1)],
                t_seq,
                t_parallel_times,
                t_parallel_times_shared,
                t_parallel_times_joblib,
                t_parallel_times_joblib_shared
            )

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

            speedup_joblib = t_seq / t_par_joblib
            print(f"\n SPEEDUP Joblib: {speedup_joblib:.2f}x")
            if speedup_joblib > 1:
                print(f"   (Il parallelo Joblib è {speedup_joblib:.2f} volte più veloce)")
            else:
                print("   (Il parallelo Joblib è più lento: overhead > guadagno)")

            speedup_joblib_shared = t_seq / t_par_joblib_shared
            print(f"\n SPEEDUP Joblib Shared Mem: {speedup_joblib_shared:.2f}x")
            if speedup_joblib_shared > 1:
                print(f"   (Il parallelo Joblib Shared Mem è {speedup_joblib_shared:.2f} volte più veloce)")
            else:
                print("   (Il parallelo Joblib Shared Mem è più lento: overhead > guadagno)")


            compare_performance(bf_seq, bf_par, bf_par_shared,bf_joblib, bf_joblib_shared, dataset)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

