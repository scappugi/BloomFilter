import concurrent.futures
import csv
import time
import sys
import os
import numpy as np
import EmailManager
import BloomFilter

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

    local_bytearray = bytearray(m)

    for raw_email in emails_chunk:
        email = local_em.normalize_email(raw_email)
        indices = BloomFilter.BloomFilter.calculate_hashes(email, m, k)

        for idx in indices:
            local_bytearray[idx] = 1

    return local_bytearray


def main():
    print(f"--- BLOOM FILTER PY {sys.version.split()[0]} + BYTEARRAY ---")

    gil_status = "DISATTIVATO" if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled() else "ATTIVO"
    print(f"Stato GIL: {gil_status}")

    filename = "dataset_10m.csv"
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
        results = list(executor.map(worker_thread, chunks))

        print("Merge vettorizzato...", end=" ", flush=True)

        # Creiamo un accumulatore NumPy
        final_array = np.zeros(dummy_bf.m, dtype=np.uint8)

        for ba in results:
            arr_view = np.frombuffer(ba, dtype=np.uint8)
            np.bitwise_or(final_array, arr_view, out=final_array)

        print("Fatto.")

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\nTempo impiegato: {elapsed:.4f} secondi")
    print(f"Velocità: {real_n_emails / elapsed:.0f} email/sec")
    print(f"Bit a 1: {np.sum(final_array)}")


if __name__ == "__main__":
    main()