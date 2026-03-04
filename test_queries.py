import time
from time import perf_counter
import EmailManager
import random
from EmailFilterService import EmailFilterService
from verify_correctness import TestBloomCorrectness


def print_bloom_correctness(bloom_filter, dataset_presenti, test_assenti, em, probability):

    print("\n--- VERIFICA CORRETTEZZA (FP / FN) ---")
    false_negatives = 0
    false_positives = 0

    # 1. Verifica assenza di Falsi Negativi (sul dataset che abbiamo inserito)
    for raw_email in dataset_presenti:
        normalized = em.normalize_email(raw_email)
        if not bloom_filter.contains(normalized):
            false_negatives += 1

    if false_negatives == 0:
        print(" False Negatives: 0 (OK - Nessuna email persa)")
    else:
        print(f" GRAVE BUG: Trovati {false_negatives} Falsi Negativi! Il filtro è rotto.")

    # 2. Verifica tasso di Falsi Positivi (sul dataset di elementi non presenti)
    for raw_email in test_assenti:
        normalized = em.normalize_email(raw_email)
        if bloom_filter.contains(normalized):
            false_positives += 1

    totale_assenti = len(test_assenti)
    measured_fpr = false_positives / totale_assenti

    print(f"Falsi Positivi trovati: {false_positives} su {totale_assenti} testati")
    print(f"FPR Misurato: {measured_fpr:.4f} (Target teorico: {probability})")

    tolerance = 0.015
    if measured_fpr <= probability + tolerance:
        print("FPR entro la tolleranza prevista.")
    else:
        print(f"ATTENZIONE: FPR troppo alto! Atteso ~{probability}, ottenuto {measured_fpr}")


def run_query_benchmark():
    N_TRAIN = 500000
    N_TEST = 100000
    PROBABILITY = 0.01

    print("1. Generazione Dati...")
    service = EmailFilterService(N_TRAIN, PROBABILITY)
    em = EmailManager.EmailManager()  # Usato solo qui per generare i dataset di test

    dataset_training = em.generate_complex_email(N_TRAIN)

    print("2. Popolamento del BloomFilter in corso...")
    for raw_email in dataset_training:
        service.add_email(raw_email)

    print("3. Preparazione dei dataset di test...")
    test_presenti = random.sample(dataset_training, N_TEST)

    test_assenti = []
    #uso set per velocizzare la ricerca da O(n) a O(1)
    dataset_set = set(dataset_training)
    #genero il doppio delle email richieste
    candidates = em.generate_complex_email(N_TEST * 2)
    for c in candidates:
        if c not in dataset_set:
            test_assenti.append(c)
        #uscita anticipata nel caso in cui raggiungo la quota prima della fine del ciclo
        if len(test_assenti) == N_TEST:
            break


    print("\n--- AVVIO BENCHMARK SISTEMA COMPLETO ---")

    # Test Presenti
    start_time = perf_counter()
    for email in test_presenti:
        service.is_email_present(email)
    tempo_presenti = perf_counter() - start_time

    # Test Assenti
    start_time = perf_counter()
    for email in test_assenti:
        service.is_email_present(email)
    tempo_assenti = perf_counter() - start_time

    # Calcolo metriche
    lat_presenti = (tempo_presenti / N_TEST) * 1_000_000
    lat_assenti = (tempo_assenti / N_TEST) * 1_000_000

    print(f"A) Elementi PRESENTI:")
    print(f"   Latenza media: {lat_presenti:.2f} µs/query")
    print(f"   Throughput:    {N_TEST / tempo_presenti:,.0f} query/sec")

    print(f"\nB) Elementi ASSENTI:")
    print(f"   Latenza media: {lat_assenti:.2f} µs/query")
    print(f"   Throughput:    {N_TEST / tempo_assenti:,.0f} query/sec")
    print_bloom_correctness(
        bloom_filter=service.bloom,
        dataset_presenti=dataset_training,
        test_assenti=test_assenti,
        em=service.em,
        probability=PROBABILITY
    )




if __name__ == "__main__":
    run_query_benchmark()