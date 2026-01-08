import concurrent.futures
import csv
import time
import sys
import os
import numpy as np
from bitarray import bitarray

import EmailManager
import BloomFilter
from test import run_sequential

PROBABILITY = 0.01
_bf_params = None  # (m, k)
_em = EmailManager.EmailManager()


def load_dataset_from_csv(filename):
    if not os.path.exists(filename):
        print(f" ERRORE: Il file {filename} non esiste.")
        return None
    print(f"Caricamento {filename}...", end="", flush=True)
    dataset = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row: dataset.append(row[0])
    print(f" Fatto. ({len(dataset)} email)")
    return dataset


def worker_thread(emails_chunk):
    m, k = _bf_params
    local_em = _em

    local_bytearray = bitarray(m)
    local_bytearray.setall(0)

    for raw_email in emails_chunk:
        email = local_em.normalize_email(raw_email)
        indices = BloomFilter.BloomFilter.calculate_hashes(email, m, k)

        for idx in indices:
            local_bytearray[idx] = 1

    return local_bytearray


def main():
    dt1 = load_dataset_from_csv("dataset_500k.csv")
    bf_seq, t_seq = run_sequential(dt1, len(dt1), PROBABILITY)

    print(f"--- BLOOM FILTER PY {sys.version.split()[0]} + BYTEARRAY ---")

    gil_status = "DISATTIVATO" if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled() else "ATTIVO"
    print(f"Stato GIL: {gil_status}")

    filename = "dataset_500k.csv"
    dataset = load_dataset_from_csv(filename)
    if dataset is None: return

    real_n_emails = len(dataset)

    # Calcolo parametri BF
    dummy_bf = BloomFilter.BloomFilter.from_probability(real_n_emails, PROBABILITY)
    global _bf_params
    _bf_params = (dummy_bf.m, dummy_bf.k)

    print(f"Parametri: m={dummy_bf.m}, k={dummy_bf.k}")
    print(f"RAM per thread (bytearray): {dummy_bf.m / 1024 / 1024:.2f} MB")

    num_threads = os.cpu_count()
    print(f"Avvio {num_threads} thread...")

    chunk_size = len(dataset) // num_threads
    chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(worker_thread, chunks)

        print("Merge vettorizzato...", end=" ", flush=True)

        buffer = bitarray(dummy_bf.m)
        buffer.setall(0)

        # passo di reduce
        for ba in results:
            buffer |= ba  # bitarray supporta l'operatore OR direttamente

        dummy_bf.bit_array = buffer
        print("Fatto.")

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\nTempo impiegato: {elapsed:.4f} secondi")
    print(f"Velocità: {real_n_emails / elapsed:.0f} email/sec")
    print(f"Bit a 1: {np.sum(dummy_bf.bit_array.count())}")
    print(f"lunghezza bitarray: {dummy_bf.m}")


if __name__ == "__main__":
    main()