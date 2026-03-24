import os
import sys
import time
import concurrent.futures
import multiprocessing
from multiprocessing import shared_memory
import numpy as np

# Assicuriamoci che i percorsi siano corretti
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import EmailManager, BloomFilter
import random


def worker_thread_profiled(bloom_filter, em, emails):
    """Worker Thread: Ritorna anche il tempo di esecuzione interno."""
    t0 = time.perf_counter()
    count = 0
    for email in emails:
        if bloom_filter.contains(em.normalize_email(email)):
            count += 1
    t1 = time.perf_counter()
    return count, (t1 - t0)


def worker_process_profiled(shm_name, em, m, k, emails):
    t_start = time.perf_counter()

    # Misuriamo il tempo per connettersi alla memoria condivisa (Overhead IPC locale)
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(shape=(m,), dtype=np.int8, buffer=shm.buf)

    t_ready = time.perf_counter()
    count = 0

    try:
        # CORE WORK (Calcolo utile)
        for raw_email in emails:
            email = em.normalize_email(raw_email)
            indices = BloomFilter.BloomFilter.calculate_hashes(email, m, k)
            presente = True
            for idx in indices:
                if shared_array[idx] == 0:
                    presente = False
                    break
            if presente:
                count += 1
    finally:
        shm.close()

    t_end = time.perf_counter()

    # risultato, tempo di connessione SHM, tempo utile di calcolo
    return count, (t_ready - t_start), (t_end - t_ready)


def run_profiling():
    N_TRAIN = 1000000
    N_TEST = 200000
    WORKERS = 4
    PROBABILITY = 0.01

    print(f"Preparazione Dati ({N_TRAIN} Train, {N_TEST} Test)...")
    em = EmailManager.EmailManager()
    bf = BloomFilter.BloomFilter.from_probability(N_TRAIN, PROBABILITY)
    dataset_training = em.generate_complex_email(N_TRAIN)
    for email in dataset_training:
        bf.add(em.normalize_email(email))

    test_presenti = random.sample(dataset_training, N_TEST)

    def split_list(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    chunks = split_list(test_presenti, WORKERS)

    print("\n" + "=" * 55)
    print(f" PROFILAZIONE THREAD (No-GIL) - {WORKERS} Worker")
    print("=" * 55)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        t1 = time.perf_counter()
        futures = [executor.submit(worker_thread_profiled, bf, em, chunk) for chunk in chunks]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        t2 = time.perf_counter()

    core_times = [res[1] for res in results]
    max_core_time = max(core_times)  # Il tempo che ci mette il thread più lento

    print(f"1. Setup & Lancio Pool:       {(t1 - t0) * 1000:8.2f} ms")
    print(f"2. Esecuzione Reale (Core):   {max_core_time * 1000:8.2f} ms  <-- Tempo speso a fare HASH")
    print(f"3. Overhead (Attesa/Sync):    {((t2 - t1) - max_core_time) * 1000:8.2f} ms")
    print(f"-" * 55)
    print(f"TEMPO TOTALE (Wall Time):     {(t2 - t0) * 1000:8.2f} ms")

    print("\n" + "=" * 55)
    print(f" PROFILAZIONE PROCESS (Shared Mem) - {WORKERS} Worker")
    print("=" * 55)

    t0 = time.perf_counter()
    m = bf.get_size()
    k = bf.get_hash_count()

    # FASE 1: Creazione e Copia in Memoria Condivisa (Il grande scoglio)
    size_in_bytes = m * np.dtype(np.int8).itemsize
    shm = shared_memory.SharedMemory(create=True, size=size_in_bytes)
    try:
        bit_array = np.ndarray(shape=(m,), dtype=np.int8, buffer=shm.buf)
        bit_array.fill(0)
        bit_array[:] = bf.bit_array.tolist()  # COPIA COSTOSISSIMA
        t1 = time.perf_counter()

        # FASE 2: Esecuzione
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
            t2 = time.perf_counter()
            futures = [executor.submit(worker_process_profiled, shm.name, em, m, k, chunk) for chunk in chunks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            t3 = time.perf_counter()

    finally:
        shm.close()
        shm.unlink()

    conn_times = [res[1] for res in results]
    core_times = [res[2] for res in results]
    max_core_time = max(core_times)
    avg_conn_time = sum(conn_times) / len(conn_times)

    print(f"1. Creazione SHM & Copia:     {(t1 - t0) * 1000:8.2f} ms  <-- TEMPO PERSO IN IPC")
    print(f"2. Lancio Process Pool:       {(t2 - t1) * 1000:8.2f} ms")
    print(f"3. Connessione Worker a SHM:  {avg_conn_time * 1000:8.2f} ms  (Media per worker)")
    print(f"4. Esecuzione Reale (Core):   {max_core_time * 1000:8.2f} ms  <-- Tempo speso a fare HASH")
    print(f"5. Overhead (Ser/Deser/Sync): {((t3 - t2) - max_core_time) * 1000:8.2f} ms")
    print(f"-" * 55)
    print(f"TEMPO TOTALE (Wall Time):     {(t3 - t0) * 1000:8.2f} ms")
    print("=" * 55)


if __name__ == "__main__":
    run_profiling()