import os
import time
import multiprocessing
import sys
import matplotlib.pyplot as plt
import csv

import BloomFilter
import EmailManager
import orchestrator

# Importiamo alcune utility da test.py per evitare duplicazioni
from test import load_dataset_from_csv, run_sequential, evaluate_filter

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
        # print(f".", end="", flush=True)
        start = time.time()

        # Chiama il metodo threaded implementato nell'orchestrator
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
        # Nota: run_threaded_worker_bytearray non ritorna il BF, ma aggiorna self.bloom
        # Dobbiamo assicurarci che orchestrator ritorni self.bloom o accedervi dopo
        orch.run_threaded_worker_bytearray(dataset)
        bf = orch.bloom
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


def plot_scalability_threaded(dataset_name, worker_counts, seq_time, thread_merge_times, thread_byte_times, thread_shared_times):
    """
    Versione aggiornata per 3 curve.
    """
    plt.figure(figsize=(12, 5))

    # --- Grafico 1: Tempi Assoluti ---
    plt.subplot(1, 2, 1)
    plt.axhline(y=seq_time, color='r', linestyle='-', label=f"Sequenziale ({seq_time:.2f}s)")
    
    plt.plot(worker_counts, thread_merge_times, marker='o', label="Merge (Bitarray)")
    plt.plot(worker_counts, thread_byte_times, marker='^', label="Merge (ByteArray)")
    plt.plot(worker_counts, thread_shared_times, marker='s', label="Shared (NumPy)")

    plt.title(f"Tempi Threading: {dataset_name}")
    plt.xlabel("Numero di Thread")
    plt.ylabel("Tempo (s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # --- Grafico 2: Speedup ---
    plt.subplot(1, 2, 2)
    
    sp_merge = [seq_time / t for t in thread_merge_times]
    sp_byte = [seq_time / t for t in thread_byte_times]
    sp_shared = [seq_time / t for t in thread_shared_times]

    plt.plot(worker_counts, sp_merge, marker='o', label="Bitarray")
    plt.plot(worker_counts, sp_byte, marker='^', label="ByteArray")
    plt.plot(worker_counts, sp_shared, marker='s', label="Shared (NumPy)")
    
    # Linea ideale
    plt.plot(worker_counts, worker_counts, 'k--', alpha=0.3, label="Ideale")
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)

    plt.title(f"Speedup Threading: {dataset_name}")
    plt.xlabel("Numero di Thread")
    plt.ylabel("Speedup (Nx)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


def compare_performance_threaded(bf_seq, bf_merge, bf_byte, bf_shared, training_dataset, test_size=10000):
    """
    Confronta l'accuratezza dei 4 filtri (Seq + 3 Threaded).
    """
    print(f"\n--- CONFRONTO ACCURATEZZA (Test su {test_size} email) ---")
    
    dataset_set = set(training_dataset)
    em = EmailManager.EmailManager()
    test_emails = em.generate_complex_email(test_size)

    # 1. Sequenziale
    tp_seq, fp_seq = evaluate_filter(bf_seq, test_emails, dataset_set, em)
    
    # 2. Threaded Merge (Bitarray)
    tp_merge, fp_merge = evaluate_filter(bf_merge, test_emails, dataset_set, em)

    # 3. Threaded Merge (ByteArray)
    tp_byte, fp_byte = evaluate_filter(bf_byte, test_emails, dataset_set, em)
    
    # 4. Threaded Shared
    tp_shared, fp_shared = evaluate_filter(bf_shared, test_emails, dataset_set, em)

    # Stampa Tabella
    header_len = 100
    print("\n" + "=" * header_len)
    print(f"{'METRICA':<20} | {'SEQ':<10} | {'TH-BIT':<10} | {'TH-BYTE':<10} | {'TH-SHARED':<10}")
    print("=" * header_len)
    print(f"{'True Positives':<20} | {tp_seq:<10} | {tp_merge:<10} | {tp_byte:<10} | {tp_shared:<10}")
    print(f"{'False Positives':<20} | {fp_seq:<10} | {fp_merge:<10} | {fp_byte:<10} | {fp_shared:<10}")
    
    fpr_seq = fp_seq / test_size
    fpr_merge = fp_merge / test_size
    fpr_byte = fp_byte / test_size
    fpr_shared = fp_shared / test_size
    
    print(f"{'False Positive Rate':<20} | {fpr_seq:.4f}     | {fpr_merge:.4f}     | {fpr_byte:.4f}     | {fpr_shared:.4f}")
    print("-" * header_len)
    print(f"{'Size (m)':<20} | {bf_seq.get_size():<10} | {bf_merge.get_size():<10} | {bf_byte.get_size():<10} | {bf_shared.get_size():<10}")
    print("=" * header_len)

    if tp_seq == tp_merge == tp_byte == tp_shared and fp_seq == fp_merge == fp_byte == fp_shared:
        print("\n✅ SUCCESSO: I risultati sono matematicamente IDENTICI.")
    else:
        print("\n⚠️ ATTENZIONE: I risultati differiscono!")


def main():
    # Controllo stato GIL
    gil_status = "DISATTIVATO" if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled() else "ATTIVO"
    print(f"--- BENCHMARK THREADING (GIL: {gil_status}) ---")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"CPU Cores: {os.cpu_count()}")

    PROBABILITY = 0.01
    DATASETS_FILES = ["dataset_10k.csv", "dataset_100k.csv", "dataset_500k.csv", 
                      "dataset_1.5m.csv", "dataset_3m.csv", "dataset_5m.csv", "dataset_10m.csv"]

    # Test rapido iniziale
    dt_name = "dataset_500k.csv"
    if os.path.exists(dt_name):
        print(f"\n--- Test Preliminare su {dt_name} ---")
        dt = load_dataset_from_csv(dt_name)
        bf_seq, t_seq = run_sequential(dt, len(dt), PROBABILITY)
        bf_th, t_th = run_threaded_shared(dt, len(dt), PROBABILITY, 4)
        bf_byte, t_byte = run_threaded_bytearray(dt, len(dt), PROBABILITY, 4)
        bf_merge, t_merge = run_threaded_merge(dt, len(dt), PROBABILITY, 4)
        print(f"Speedup Preliminare (4 thread): {t_seq/t_th:.2f}x")

        if bf_seq.bit_array == bf_th.bit_array == bf_byte.bit_array == bf_merge.bit_array:
            print("✅ I bitarray sono IDENTICI.")
        else:
            print("❌ I bitarray sono DIVERSI.")


    # Loop completo sui dataset
    for filename in DATASETS_FILES:
        dataset = load_dataset_from_csv(filename)
        if dataset is None: continue

        print("\n" + "=" * 60)
        print(f" DATASET: {filename} ({len(dataset)} email)")
        print("=" * 60)

        N_EMAILS = len(dataset)
        
        # Baseline Sequenziale
        bf_seq, t_seq = run_sequential(dataset, N_EMAILS, PROBABILITY)

        times_merge = []
        times_byte = []
        times_shared = []
        
        # Test Scalabilità Thread (da 1 a 2x CPU)
        max_threads = os.cpu_count() * 2
        thread_counts = range(1, max_threads + 1)

        bf_merge_final = None
        bf_byte_final = None
        bf_shared_final = None

        for n_thread in thread_counts:
            print(f"\n--- Test con {n_thread} Thread ---")
            
            # 1. Threaded Merge (Bitarray)
            bf_merge, t_merge = run_threaded_merge(dataset, N_EMAILS, PROBABILITY, n_thread)
            times_merge.append(t_merge)
            bf_merge_final = bf_merge

            # 2. Threaded Merge (ByteArray)
            bf_byte, t_byte = run_threaded_bytearray(dataset, N_EMAILS, PROBABILITY, n_thread)
            times_byte.append(t_byte)
            bf_byte_final = bf_byte

            # 3. Threaded Shared (NumPy)
            bf_shared, t_shared = run_threaded_shared(dataset, N_EMAILS, PROBABILITY, n_thread)
            times_shared.append(t_shared)
            bf_shared_final = bf_shared

        # Calcolo Speedup massimi
        best_time_merge = min(times_merge)
        best_time_byte = min(times_byte)
        best_time_shared = min(times_shared)
        
        print(f"\nRISULTATI PER {filename}:")
        print(f"Speedup Max Merge (Bit):  {t_seq / best_time_merge:.2f}x")
        print(f"Speedup Max Merge (Byte): {t_seq / best_time_byte:.2f}x")
        print(f"Speedup Max Shared:       {t_seq / best_time_shared:.2f}x")

        # Grafico
        plot_scalability_threaded(filename, list(thread_counts), t_seq, times_merge, times_byte, times_shared)

        # Verifica Correttezza (usando l'ultimo BF generato)
        compare_performance_threaded(bf_seq, bf_merge_final, bf_byte_final, bf_shared_final, dataset)

if __name__ == "__main__":
    main()
