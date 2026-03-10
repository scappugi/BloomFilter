import multiprocessing
from src import orchestrator, EmailManager


def main():
    N_EMAILS = 500000
    PROBABILITY = 0.01

    print("Generazione dataset in corso...")
    em = EmailManager.EmailManager()
    dataset = em.generate_complex_email(N_EMAILS)
    with open("stuff/emails_dataset.csv", "w", encoding="utf-8") as f:
        for email in dataset:
            f.write(email + "\n")
    print("Dataset generato.")

    print(f"Avvio MapReduce con {multiprocessing.cpu_count()} worker...")

    o = orchestrator.BloomOrchestrator(N_EMAILS, PROBABILITY)

    bloom_filter = o.process_chunks(dataset)
    print("MapReduce completato.")

    # Generazione test set casuale e conteggio TP/FP
    test_size = 10000
    true_positives = 0
    false_positives = 0

    emailManager = EmailManager.EmailManager()
    for _ in range(test_size):
        email = emailManager.normalize_email(emailManager.generate_complex_email(1)[0])

        if bloom_filter.contains(email):
            if email in dataset:
                true_positives += 1
            else:
                false_positives += 1

    print(f"Finito! Bit impostati a 1 nel filtro.")

    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False positive rate: {false_positives / test_size:.4f}")

    # Parametri del Bloom Filter
    print(f"Size (m): {bloom_filter.get_size()}")
    print(f"Hash functions (k): {bloom_filter.get_hash_count()}")
    print(f"False positive rate (p): {bloom_filter.get_false_positive_rate()}")





if __name__ == "__main__":
    main()