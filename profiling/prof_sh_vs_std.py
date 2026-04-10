import os
import sys

# Assicuriamoci che i percorsi siano corretti
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests import test_utils
from src import orchestrator


def run_timer_bench():
    # Usiamo 100k (o 500k) per dare ai worker abbastanza lavoro da misurare
    filename = "dataset_1.5m.csv"
    print(f"Caricamento {filename} in memoria...")
    dataset = test_utils.load_dataset_from_csv(filename)

    if not dataset:
        return

    n = len(dataset)
    p = 0.01  # Probabilità
    WORKERS = 4  # Numero di core che si daranno "battaglia"

    # Inizializziamo l'orchestratore
    orch = orchestrator.BloomOrchestrator(n, p, num_workers=WORKERS)

    print(f"\n" + "=" * 50)
    print(f" ESECUZIONE STANDARD (Array Locali) - {WORKERS} Worker")
    print("=" * 50)
    # Questa chiamata scatenerà i 4 worker, e ognuno stamperà il suo tempo
    orch.process_chunks(dataset)

    print(f"\n" + "=" * 50)
    print(f" ESECUZIONE SHARED (Memoria Condivisa) - {WORKERS} Worker")
    print("=" * 50)
    # Anche qui, 4 worker partiranno e stamperanno i tempi
    orch.run_worker(dataset)


if __name__ == "__main__":
    run_timer_bench()