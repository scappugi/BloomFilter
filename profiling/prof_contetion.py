import os
import sys
import time
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt
from tests import test_utils

# Aggiunta del percorso radice per gli import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import orchestrator, BloomFilter, EmailManager


### NON HA FORNITO RISULTATI UTILI -> NON è STATO COINVOLTO NELL' ANALISI

def run_contention_profiling():
    N_EMAILS = 1500000
    P = 0.01
    #worker al numero di core disponibili
    WORKERS = multiprocessing.cpu_count()
    print("WORKERS:", WORKERS)
    orch = orchestrator.BloomOrchestrator(N_EMAILS, P, num_workers=WORKERS)
    em = EmailManager.EmailManager()

    #dataset normale con 500k email
    filename = "dataset_1.5m.csv"
    print(f"Caricamento dataset {filename} in memoria...")
    dataset = test_utils.load_dataset_from_csv(filename)

    print(f"Generazione del dataset patologico (alta contesa)...")
    # Generiamo un dataset con molte email duplicate per aumentare la contesa
    single_email = "contention_test@example.com"
    dataset_pathological = [single_email] * N_EMAILS

    print("\nProfilazione con dataset normale (bassa contesa)...")
    start_time = time.perf_counter()
    orch.run_threaded_shared(dataset_pathological)
    end_time = time.perf_counter()
    print(f"Tempo impiegato (dataset normale): {end_time - start_time:.4f}s")

    print("\nProfilazione con dataset patologico (alta contesa)...")
    start_time = time.perf_counter()
    orch.run_threaded_shared(dataset_pathological)
    end_time = time.perf_counter()
    print(f"Tempo impiegato (dataset patologico): {end_time - start_time:.4f}s")

    #Test con processi
    print("\nProfilazione con dataset normale (bassa contesa) - Processi...")
    start_time = time.perf_counter()
    orch.run_worker(dataset_pathological)
    end_time = time.perf_counter()
    print(f"Tempo impiegato (dataset normale - processi): {end_time - start_time:.4f}s")

    print("\nProfilazione con dataset patologico (alta contesa) - Processi...")
    start_time = time.perf_counter()
    orch.run_worker(dataset_pathological)
    end_time = time.perf_counter()
    print(f"Tempo impiegato (dataset patologico - processi): {end_time - start_time:.4f}s")

if __name__ == "__main__":
    run_contention_profiling()



