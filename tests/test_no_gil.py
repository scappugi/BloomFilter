import os
import sys
import time
import multiprocessing

# --- Inizio Blocco di Correzione Percorsi ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fine Blocco ---

from src import orchestrator
from tests import test_utils
from tests.test import run_sequential # Manteniamo la baseline sequenziale

def run_threaded_merge(dataset, n, p, n_threads):
    """
    Esegue il Bloom Filter usando ThreadPoolExecutor con strategia Map-Reduce (Merge dei Bitarray).
    """
    print(f"\n[Threaded Merge (Bitarray)] Avvio con {n_threads} thread...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, num_workers=n_threads)
    n_runs = 5
    avg_times = []
    bf = None
    for i in range(n_runs):
        start = time.time()
        bf = orch.run_threaded_worker(dataset)
        end = time.time()
        avg_times.append(end - start)
    avg_time = sum(avg_times) / len(avg_times)
    print(f" Tempo medio: {avg_time:.4f}s")
    return bf, avg_time

def run_threaded_bytearray(dataset, n, p, n_threads):
    """
    Esegue il Bloom Filter usando ThreadPoolExecutor con strategia Map-Reduce (Merge dei ByteArray).
    """
    print(f"\n[Threaded Merge (ByteArray)] Avvio con {n_threads} thread...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, num_workers=n_threads)
    n_runs = 5
    avg_times = []
    bf = None
    for i in range(n_runs):
        start = time.time()
        bf = orch.run_threaded_worker_bytearray(dataset)
        end = time.time()
        avg_times.append(end - start)
    avg_time = sum(avg_times) / len(avg_times)
    print(f" Tempo medio: {avg_time:.4f}s")
    return bf, avg_time

def run_threaded_shared(dataset, n, p, n_threads):
    """
    Esegue il Bloom Filter usando ThreadPoolExecutor con strategia Shared Memory (NumPy Array).
    """
    print(f"\n[Threaded Shared (NumPy)]   Avvio con {n_threads} thread...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, num_workers=n_threads)
    n_runs = 5
    avg_times = []
    bf = None
    for i in range(n_runs):
        start = time.time()
        bf = orch.run_threaded_shared(dataset)
        end = time.time()
        avg_times.append(end - start)
    avg_time = sum(avg_times) / len(avg_times)
    print(f" Tempo medio: {avg_time:.4f}s")
    return bf, avg_time

def main():
    # Controllo stato GIL
    gil_status = "DISATTIVATO" if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled() else "ATTIVO"
    print(f"--- BENCHMARK THREADING (GIL: {gil_status}) ---")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"CPU Cores: {os.cpu_count()}")

    PROBABILITY = 0.01
    DATASETS_FILES = ["dataset_10k.csv", "dataset_100k.csv", "dataset_500k.csv", 
                      "dataset_1.5m.csv", "dataset_3m.csv", "dataset_5m.csv", "dataset_10m.csv"]

    # Loop completo sui dataset
    for filename in DATASETS_FILES:
        dataset = test_utils.load_dataset_from_csv(filename)
        if dataset is None: continue

        print("\n" + "=" * 60)
        print(f" DATASET: {filename} ({len(dataset)} email)")
        print("=" * 60)

        N_EMAILS = len(dataset)
        
        # Baseline Sequenziale
        bf_seq, t_seq = run_sequential(dataset, N_EMAILS, PROBABILITY)

        # Dizionari per raccogliere i risultati
        results_times = {}
        filters = {'SEQ': bf_seq}

        # Liste temporanee
        times_merge = []
        times_byte = []
        times_shared = []
        
        max_threads = os.cpu_count() * 2
        thread_counts = range(1, max_threads + 1)

        for n_thread in thread_counts:
            print(f"\n--- Test con {n_thread} Thread ---")
            
            bf_merge, t_merge = run_threaded_merge(dataset, N_EMAILS, PROBABILITY, n_thread)
            times_merge.append(t_merge)
            filters['TH-Bit'] = bf_merge

            bf_byte, t_byte = run_threaded_bytearray(dataset, N_EMAILS, PROBABILITY, n_thread)
            times_byte.append(t_byte)
            filters['TH-Byte'] = bf_byte

            bf_shared, t_shared = run_threaded_shared(dataset, N_EMAILS, PROBABILITY, n_thread)
            times_shared.append(t_shared)
            filters['TH-Shared'] = bf_shared

        results_times['Merge (Bitarray)'] = times_merge
        results_times['Merge (ByteArray)'] = times_byte
        results_times['Shared (NumPy)'] = times_shared
        
        print(f"\nGenerazione grafico scalabilità per {filename}...")
        test_utils.plot_scalability(filename, list(thread_counts), t_seq, results_times)

        test_utils.compare_performance(filters, dataset)

if __name__ == "__main__":
    main()
