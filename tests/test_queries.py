import time
from time import perf_counter
from src import EmailManager
from src import BloomFilter
import random
import concurrent.futures
import multiprocessing

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


def worker_query(bloom_filter, em, emails):
    #cuore del codice parallelo, riceve un chunk e verifica la presenza delle email.
    count = 0
    for email in emails:
        normalized = em.normalize_email(email)
        if bloom_filter.contains(normalized):
            count += 1
    return count


def run_query_benchmark_parallel(bloom_filter, em, test_presenti, test_assenti, n_threads=None):
    if n_threads is None:
        n_threads = multiprocessing.cpu_count()

    print(f"\n--- AVVIO BENCHMARK PARALLELO ({n_threads} Thread) ---")
    
    # dividere in chunk
    def split_list(lst, n):
        k, m = divmod(len(lst), n)
        chunks = []
        start = 0
        for i in range(n):
            end = start + k + (1 if i < m else 0)
            chunks.append(lst[start:end])
            start = end
        return chunks

    chunks_presenti = split_list(test_presenti, n_threads)
    chunks_assenti = split_list(test_assenti, n_threads)

    # Test Presenti
    start_time = perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Eseguiamo le query in parallelo
        futures = [executor.submit(worker_query, bloom_filter, em, chunk) for chunk in chunks_presenti]
        # Attendiamo i risultati (opzionale se ci interessa solo il tempo, ma utile per correttezza)
        results_presenti = sum(f.result() for f in concurrent.futures.as_completed(futures))
    
    tempo_presenti = perf_counter() - start_time

    # Test Assenti
    start_time = perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker_query, bloom_filter, em, chunk) for chunk in chunks_assenti]
        results_assenti = sum(f.result() for f in concurrent.futures.as_completed(futures))
    
    tempo_assenti = perf_counter() - start_time

    N_TEST = len(test_presenti) # Assumiamo len(test_presenti) == len(test_assenti) per semplicità di stampa

    # Calcolo metriche
    lat_presenti = (tempo_presenti / N_TEST) * 1_000_000
    lat_assenti = (tempo_assenti / N_TEST) * 1_000_000

    print(f"A) Elementi PRESENTI (Parallelo):")
    print(f"   Latenza media (wall-clock): {lat_presenti:.2f} µs/query")
    print(f"   Throughput:    {N_TEST / tempo_presenti:,.0f} query/sec")
    print(f"   Trovati:       {results_presenti}/{N_TEST}")

    print(f"\nB) Elementi ASSENTI (Parallelo):")
    print(f"   Latenza media (wall-clock): {lat_assenti:.2f} µs/query")
    print(f"   Throughput:    {N_TEST / tempo_assenti:,.0f} query/sec")
    print(f"   Trovati (FP):  {results_assenti}/{N_TEST}")


def run_query_benchmark():
    N_TRAIN = 500000
    N_TEST = 100000
    PROBABILITY = 0.01

    print("1. Generazione Dati...")
    # Sostituito EmailFilterService con uso diretto
    em = EmailManager.EmailManager()
    bf = BloomFilter.BloomFilter.from_probability(N_TRAIN, PROBABILITY)

    dataset_training = em.generate_complex_email(N_TRAIN)

    print("2. Popolamento del BloomFilter in corso...")
    for raw_email in dataset_training:
        normalized = em.normalize_email(raw_email)
        bf.add(normalized)

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


    print("\n--- AVVIO BENCHMARK SISTEMA COMPLETO (Sequenziale) ---")

    # Test Presenti
    start_time = perf_counter()
    for email in test_presenti:
        normalized = em.normalize_email(email)
        bf.contains(normalized)
    tempo_presenti = perf_counter() - start_time

    # Test Assenti
    start_time = perf_counter()
    for email in test_assenti:
        normalized = em.normalize_email(email)
        bf.contains(normalized)
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
        bloom_filter=bf,
        dataset_presenti=dataset_training,
        test_assenti=test_assenti,
        em=em,
        probability=PROBABILITY
    )

    run_query_benchmark_parallel(bf, em, test_presenti, test_assenti)
    
    # Analisi qualitativa dei Falsi Positivi
    analyze_false_positives_similarity(bf, em, dataset_training, test_assenti)


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def analyze_false_positives_similarity(bloom_filter, em, dataset_training, test_assenti, sample_size=5):
    print(f"\n--- ANALISI QUALITATIVA FALSI POSITIVI (Campione di {sample_size}) ---")
    
    # 1. Troviamo i Falsi Positivi
    false_positives = []
    for email in test_assenti:
        normalized = em.normalize_email(email)
        if bloom_filter.contains(normalized):
            false_positives.append(email)
            if len(false_positives) >= sample_size:
                break
    
    if not false_positives:
        print("Nessun Falso Positivo trovato da analizzare.")
        return

    print(f"Trovati {len(false_positives)} FP da analizzare. Calcolo distanza minima dal training set...")
    print("(Questo potrebbe richiedere tempo se il training set è grande)")
    training_subset = dataset_training

    for fp_email in false_positives:
        min_dist = float('inf')
        closest_email = ""
        
        for train_email in training_subset:
            dist = levenshtein_distance(fp_email, train_email)
            if dist < min_dist:
                min_dist = dist
                closest_email = train_email
                if min_dist == 1: break # Distanza minima possibile (a parte 0), inutile cercare oltre
        
        print(f"\nFP: '{fp_email}'")
        print(f"   -> Più simile: '{closest_email}' (Distanza: {min_dist})")
        
        if min_dist <= 2:
            print("   -> DIAGNOSI: Molto simile. Probabile collisione dovuta alla struttura dell'hash su input simili.")
        else:
            print("   -> DIAGNOSI: Distante. Probabile collisione casuale pura.")


if __name__ == "__main__":
    run_query_benchmark()
