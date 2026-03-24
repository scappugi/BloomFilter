import os
import sys
import time
import multiprocessing
import random

# --- Inizio Blocco di Correzione Percorsi ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fine Blocco ---

from src import orchestrator, EmailManager
from tests import test_utils
from tests.test import run_sequential


def run_static_bench(dataset, n, p, n_workers, n_runs=3):
    """Esegue il benchmark statico (process_chunks) e calcola la media."""
    print(f"    [Statico] Test con {n_workers} Worker...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, num_workers=n_workers)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        orch.process_chunks(dataset)
        times.append(time.perf_counter() - start)
    avg = sum(times) / n_runs
    print(f" Media: {avg:.4f}s")
    return avg


def run_queue_bench(dataset, n, p, n_workers, chunk_size, n_runs=3):
    """Esegue il benchmark dinamico con Manager.Queue e calcola la media."""
    print(f"    [Queue] Test con {n_workers} Worker...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, num_workers=n_workers)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        orch.process_dynamic(dataset, chunk_size=chunk_size)
        times.append(time.perf_counter() - start)
    avg = sum(times) / n_runs
    print(f" Media: {avg:.4f}s")
    return avg


def run_imap_bench(dataset, n, p, n_workers, chunk_size, n_runs=3):
    """Esegue il benchmark dinamico con imap_unordered e calcola la media."""
    print(f"    [imap] Test con {n_workers} Worker...", end="", flush=True)
    orch = orchestrator.BloomOrchestrator(n, p, num_workers=n_workers)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        orch.process_dynamic_imap(dataset, chunk_size=chunk_size)
        times.append(time.perf_counter() - start)
    avg = sum(times) / n_runs
    print(f" Media: {avg:.4f}s")
    return avg


def main():
    N_EMAILS = 1500000
    PROBABILITY = 0.01
    N_RUNS = 3
    CHUNK_SIZE_DYNAMIC = 10000  # Dimensione mini-chunk per le versioni dinamiche

    print(f"--- BENCHMARK LOAD BALANCING (Dataset Sbilanciato) ---")

    em = EmailManager.EmailManager()
    print(f"[*] Generazione dataset sbilanciato...")
    base_emails = em.generate_complex_email(N_EMAILS)
    unbalanced_dataset = [e * 200 for e in base_emails[:100000]] + base_emails[100000:]

    _, t_seq = run_sequential(unbalanced_dataset, N_EMAILS, PROBABILITY)

    results_times = {
        'Statico (Standard)': [],
        'Dinamico (Queue)': [],
        'Dinamico (imap)': []
    }

    max_workers = multiprocessing.cpu_count() * 2
    worker_counts = []

    n = 1
    while n <= max_workers:
        worker_counts.append(n)
        n *= 2

    for nw in worker_counts:
        print(f"\n--- Configurazione: {nw} Worker ---")

        # Esecuzione dei tre metodi separati
        t_static = run_static_bench(unbalanced_dataset, N_EMAILS, PROBABILITY, nw, N_RUNS)
        results_times['Statico (Standard)'].append(t_static)

        t_queue = run_queue_bench(unbalanced_dataset, N_EMAILS, PROBABILITY, nw, CHUNK_SIZE_DYNAMIC, N_RUNS)
        results_times['Dinamico (Queue)'].append(t_queue)

        t_imap = run_imap_bench(unbalanced_dataset, N_EMAILS, PROBABILITY, nw, CHUNK_SIZE_DYNAMIC, N_RUNS)
        results_times['Dinamico (imap)'].append(t_imap)

    # 4. Generazione Grafici
    print(f"\n[*] Generazione grafici di scalabilità...")
    test_utils.plot_scalability("Dataset Sbilanciato (Load Balancing)",
                                list(worker_counts), t_seq, results_times)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()