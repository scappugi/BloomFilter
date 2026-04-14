import os
import sys
from tests import test_utils

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import worker, EmailManager, BloomFilter  # <-- Aggiunto l'import di BloomFilter
import multiprocessing

#!!!!!!!!!!! Per usare questo medoto devi utilizzare @profile su tutte le funzioni che vuoi profilare
def run_bench():
    filename = "dataset_500k.csv"
    print(f"Caricamento {filename} in memoria...")
    dataset = test_utils.load_dataset_from_csv(filename)
    n = len(dataset)
    print(f"Dataset caricato con successo: {n} email.")
    p = 0.01

    #calcola m e k automaticamente
    bf_temp = BloomFilter.BloomFilter.from_probability(n, p)
    m = bf_temp.get_size()
    k = bf_temp.get_hash_count()

    print(f"Parametri calcolati automaticamente per p={p}: m={m}, k={k}")

    em = EmailManager.EmailManager()
    dataset = em.generate_complex_email(n)

    # Inizializza l'array per 'm'
    shared_array = multiprocessing.Array('b', m, lock=False)
    worker.init_worker_shared(shared_array)

    print("Profilazione Memoria Standard...")
    worker.process_chunk((dataset, m, k))

    print("Profilazione Memoria Condivisa...")
    worker.process_chunk_shared((dataset, m, k))


if __name__ == "__main__":
    run_bench()