import csv
import time
import os
import EmailManager


def generate_csv(filename, num_emails):
    print(f"[*] Inizio generazione {num_emails} email per {filename}...")
    start = time.time()

    em = EmailManager.EmailManager()
    # Genera la lista in memoria
    emails = em.generate_complex_email(num_emails)

    # Scrive su CSV
    print(f"    Scrittura su disco in corso...")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["email"])  # Header
        for email in emails:
            writer.writerow([email])

    end = time.time()
    print(f"Completato in {end - start:.2f} secondi. File salvato: {filename}\n")


if __name__ == "__main__":
    # Configurazioni richieste
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

