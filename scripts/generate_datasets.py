import csv
import time
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import EmailManager

DATA_DIR = os.path.join(project_root, "data")


def generate_csv(filename, num_emails):
    # Costruisce il percorso completo
    full_path = os.path.join(DATA_DIR, filename)

    print(f"[*] Inizio generazione {num_emails} email per {full_path}...")
    start = time.time()

    em = EmailManager.EmailManager()
    # Genera la lista in memoria
    emails = em.generate_complex_email(num_emails)

    # Scrive su CSV
    print(f"    Scrittura su disco in corso...")
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["email"])  # Header
        for email in emails:
            writer.writerow([email])

    end = time.time()
    print(f"Completato in {end - start:.2f} secondi. File salvato: {full_path}\n")


if __name__ == "__main__":
    # Assicuriamoci che la cartella data esista
    if not os.path.exists(DATA_DIR):
        print(f"Creazione cartella dati: {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)

    # Configurazioni richieste (solo nomi file, senza percorso)
    datasets = [
        ("dataset_10k.csv", 10_000),
        ("dataset_100k.csv", 100_000),
        ("dataset_500k.csv", 500_000),
        ("dataset_1.5m.csv", 1_500_000),
        ("dataset_3m.csv", 3_000_000),
        ("dataset_5m.csv", 5_000_000),
        ("dataset_10m.csv", 10_000_000)
    ]

    print("--- GENERATORE DATASET ---")
    for fname, qty in datasets:
        generate_csv(fname, qty)
