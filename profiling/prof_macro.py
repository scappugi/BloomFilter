import os
import sys

# Assicuriamoci che i percorsi siano corretti
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests import test_utils
from src import orchestrator


################################################################
#Assicurati di mettere @profile sopra i metodi coinvolti
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
    WORKERS = 4  # Numero di core

    # Inizializziamo l'orchestratore
    orch = orchestrator.BloomOrchestrator(n, p, num_workers=WORKERS)

    print(f"\nProfilazione GLOBALE Standard...")
    orch.process_chunks(dataset)

    print(f"\nProfilazione GLOBALE Shared...")
    orch.run_worker(dataset)

    # print(f"\n Profilazione Joblib standard...")
    # orch.run_joblib_worker(dataset)
    #
    # print(f"\nProfilazione Joblib shared...")
    # orch.run_joblib_shared_worker(dataset)



if __name__ == "__main__":
    run_macro_profiling()