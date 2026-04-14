import os
import sys
import time
import multiprocessing
from joblib.externals.loky import get_reusable_executor

# --- Inizio Blocco di Correzione Percorsi ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fine Blocco ---

from src import BloomFilter, orchestrator, EmailManager
from tests import test_utils


def run_sequential(dataset, n, p):
    print(f"\n\n[Sequenziale] Iniziato...", end="", flush=True)
    bf = BloomFilter.BloomFilter.from_probability(n, p)
    em = EmailManager.EmailManager()

    calc_hashes = BloomFilter.BloomFilter.calculate_hashes
    m = bf.m
    k = bf.k
    bit_array = bf.bit_array

    avg_times = []
    n_runs = 3
    for _ in range(n_runs):
        start = time.perf_counter()

        for raw_email in dataset:

            email = em.normalize_email(raw_email)
            for idx in calc_hashes(email, m, k):
                bit_array[idx] = 1

        end = time.perf_counter()
        avg_times.append(end - start)

    avg_time = sum(avg_times) / len(avg_times)
    print(f"\n Tempo medio: {avg_time:.4f}s")
    return bf, avg_time

def run_parallel(dataset, n, p, np = multiprocessing.cpu_count()):
    print(f"\n\n[Parallelo]   Iniziato...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, np)
    n_runs = 3
    avg_times = []
    bf = None
    for i in range(n_runs):
        print (f"\n Esecuzione MapReduce n: {i}", end="", flush=True)
        start = time.perf_counter()
        bf = orch.process_chunks(dataset)
        end = time.perf_counter()
        avg_times.append(end - start)
    avg_time = sum(avg_times) / len(avg_times)
    print(f"\n Tempo medio: {avg_time:.4f}s")
    return bf, avg_time

def run_parallel_shared_memory(dataset, n, p, np = multiprocessing.cpu_count()):
    print(f"\n\n[Parallelo Shared Mem]   Iniziato...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, np)
    n_runs = 3
    avg_times = []
    bf = None
    for i in range(n_runs):
        print (f"\n Esecuzione Shared Mem {i}", end="", flush=True)
        start = time.perf_counter()
        bf = orch.run_worker(dataset)
        end = time.perf_counter()
        avg_times.append(end - start)
    avg_time = sum(avg_times) / len(avg_times)
    print(f"\n Tempo medio: {avg_time:.4f}s")
    return bf, avg_time

def run_parallel_joblib(dataset, n, p, np = multiprocessing.cpu_count()):
    print(f"\n\n[Parallelo Joblib]   Iniziato...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, np)
    n_runs = 3
    avg_times = []
    bf = None
    for i in range(n_runs):
        print (f"\n Esecuzione Joblib n: {i}", end="", flush=True)
        start = time.perf_counter()
        bf = orch.run_joblib_worker(dataset)
        end = time.perf_counter()
        get_reusable_executor().shutdown(wait=True)
        avg_times.append(end - start)
    avg_time = sum(avg_times) / len(avg_times)
    print(f"\n Tempo medio: {avg_time:.4f}s")

    return bf, avg_time

def run_parallel_joblib_shared(dataset, n, p, np = multiprocessing.cpu_count()):
    print(f"\n\n[Parallelo Joblib Shared Mem]   Iniziato...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, np)
    n_runs = 3
    avg_times = []
    bf = None
    for i in range(n_runs):
        print (f"\n Esecuzione Joblib Shared Mem n: {i}", end="", flush=True)
        start = time.perf_counter()
        bf = orch.run_joblib_shared_worker(dataset)
        end = time.perf_counter()

        #importante altrimenti joblib "bara": riutilizza i processi gia aperti senza aprirne di nuovi!
        get_reusable_executor().shutdown(wait=True)

        avg_times.append(end - start)
    avg_time = sum(avg_times) / len(avg_times)
    print(f"\n Tempo medio: {avg_time:.4f}s")

    return bf, avg_time

def main():
    PROBABILITY = 0.01
    DATASETS_FILES = ["dataset_10k.csv","dataset_100k.csv","dataset_500k.csv", "dataset_1.5m.csv"] #"dataset_3m.csv", "dataset_5m.csv", "dataset_10m.csv"

    print(f"--- BENCHMARK AUTOMATICO (CPU Cores: {multiprocessing.cpu_count()}) ---")

    dt1 = test_utils.load_dataset_from_csv("dataset_10k.csv")
    if dt1 is None:
        print("Impossibile eseguire il test preliminare, dataset non trovato.")
        return

    bf_seq, t_seq = run_sequential(dt1, len(dt1), PROBABILITY)
    bf_par, t_par = run_parallel(dt1, len(dt1), PROBABILITY, 8)
    print(f"lunghezza del bitarray {bf_par.bit_array.count()}")
    print(f"\n il tempo parallelo è: {t_par} e il tempo sequenziale è: {t_seq}, lo speedup è: {t_seq / t_par}")

    for filename in DATASETS_FILES:
        print("\n" + "=" * 60)
        print(f" DATASET: {filename}")
        print("=" * 60)

        dataset = test_utils.load_dataset_from_csv(filename)
        if dataset is None: continue

        N_EMAILS = len(dataset)
        
        # Dizionari per raccogliere i risultati
        results_times = {}
        filters = {}

        # Baseline Sequenziale
        bf_seq, t_seq = run_sequential(dataset, N_EMAILS, PROBABILITY)
        filters['SEQ'] = bf_seq
        
        # Liste temporanee per i tempi paralleli
        times_par = []
        times_shared = []
        times_job = []
        times_job_sh = []

        worker_counts = range(1, (multiprocessing.cpu_count()*2)+1)

        for n_process in worker_counts:
            print(f"\n--- Esecuzione con {n_process} processi ---")

            bf_par, t_par = run_parallel(dataset, N_EMAILS, PROBABILITY, n_process)
            times_par.append(t_par)
            filters['MP(Std)'] = bf_par # Salviamo l'ultimo per il confronto

            bf_par_shared, t_par_shared = run_parallel_shared_memory(dataset, N_EMAILS, PROBABILITY, n_process)
            times_shared.append(t_par_shared)
            filters['MP(Shm)'] = bf_par_shared

            bf_joblib, t_par_joblib = run_parallel_joblib(dataset, N_EMAILS, PROBABILITY, n_process)
            times_job.append(t_par_joblib)
            filters['JOB(Std)'] = bf_joblib

            bf_joblib_shared, t_par_joblib_shared = run_parallel_joblib_shared(dataset, N_EMAILS, PROBABILITY, n_process)
            times_job_sh.append(t_par_joblib_shared)
            filters['JOB(Nmp)'] = bf_joblib_shared

        # Popoliamo il dizionario per il plot
        results_times['MP Standard'] = times_par
        results_times['MP Shared'] = times_shared
        results_times['Joblib Std'] = times_job
        results_times['Joblib NumPy'] = times_job_sh

        print(f"\nGenerazione grafico scalabilità per {filename}...")
        test_utils.plot_scalability(filename, list(worker_counts), t_seq, results_times)

        # Confronto accuratezza usando l'ultimo set di filtri generati
        test_utils.compare_performance(filters, dataset)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
