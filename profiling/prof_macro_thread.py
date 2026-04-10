import os
import sys

# Assicuriamoci che i percorsi siano corretti
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests import test_utils
from src import orchestrator


################################################################
#RECUPERARE IL CODICE PER PROFILARE LE FUNZIONI SU GITHUB
################################################################
def run_macro_profiling():
    # Usiamo 500k o 1.5m per avere dati solidi
    filename = "dataset_500k.csv"
    print(f"Caricamento {filename} in memoria...")
    dataset = test_utils.load_dataset_from_csv(filename)

    if not dataset:
        return

    n = len(dataset)
    p = 0.01  # Probabilità

    # Lista dei worker su cui iterare
    worker_list = [1, 2, 4, 8, 16]

    for num_w in worker_list:
        print(f"\n" + "=" * 50)
        print(f" TEST CON {num_w} WORKER")
        print("=" * 50)

        # Inizializziamo l'orchestratore con il numero corrente di worker
        orch = orchestrator.BloomOrchestrator(n, p, num_workers=num_w)

        print(f"\n[Worker: {num_w}] Profilazione GLOBALE Standard...")
        orch.run_threaded_worker(dataset)

        print(f"\n[Worker: {num_w}] Profilazione GLOBALE Shared...")
        orch.run_threaded_shared(dataset)



if __name__ == "__main__":
    run_macro_profiling()
