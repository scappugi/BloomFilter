import os
import sys
import time
from time import perf_counter
from src import EmailManager
from src import BloomFilter
from src import worker
import random
import concurrent.futures
import multiprocessing
import sys
from multiprocessing import shared_memory
import numpy as np

def print_bloom_correctness(bloom_filter, dataset_presenti, test_assenti, em, probability):

    print("\n--- VERIFICA CORRETTEZZA (FP / FN) ---")
    false_negatives = 0
    false_positives = 0

    # Verifica assenza di Falsi Negativi (sul dataset che abbiamo inserito)
    for raw_email in dataset_presenti:
        normalized = em.normalize_email(raw_email)
        if not bloom_filter.contains(normalized):
            false_negatives += 1

    if false_negatives == 0:
        print(" False Negatives: 0 (OK - Nessuna email persa)")
    else:
        print(f" GRAVE BUG: Trovati {false_negatives} Falsi Negativi! Il filtro è rotto.")

    # Verifica tasso di Falsi Positivi (sul dataset di elementi non presenti)
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


def main():
    # Configurazione parametri
    N_TRAIN = 10000
    N_TEST = 5000
    PROBABILITY = 0.01

    print("1. Generazione Dati...")
    em = EmailManager.EmailManager()  #

    # Inizializzazione dell'Orchestratore (che crea internamente il BloomFilter)
    from src import orchestrator
    orch = orchestrator.BloomOrchestrator(N_TRAIN, PROBABILITY)

    dataset_training = em.generate_complex_email(N_TRAIN)  #

    print("2. Popolamento del BloomFilter...")
    for raw_email in dataset_training:
        # Accediamo all'oggetto bloom interno all'orchestratore per l'add iniziale
        orch.bloom.add(em.normalize_email(raw_email))

    print("3. Preparazione dei dataset di test...")
    test_presenti = random.sample(dataset_training, N_TEST)
    dataset_set = set(dataset_training)
    test_assenti = [c for c in em.generate_complex_email(N_TEST * 2) if c not in dataset_set][:N_TEST]

    # --- ESECUZIONE BENCHMARK TRAMITE ORCHESTRATOR ---

    # Baseline Sequenziale
    seq_pres, seq_ass = orch.query_sequential(test_presenti, test_assenti)

    worker_counts = [1, 2, 4, 8, 16]
    risultati_thread = {}
    risultati_process = {}

    for w in worker_counts:
        print(f"  -> Esecuzione con {w} worker...", end="\r", flush=True)

        # Test Thread (utilizzabile pienamente solo in Python 3.13+ No-GIL)
        gil_enabled = not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled()
        if gil_enabled:
            th_pres, th_ass = None, None
        else:
            # Metodo spostato in orchestrator.py
            th_pres, th_ass, err_total = orch.query_parallel(test_presenti, test_assenti, n_threads=w)
            print(f"⚠️ ERRORI(threads): attesi {len(test_presenti)}, ottenuti {err_total}")
        risultati_thread[w] = (th_pres, th_ass)

        # Test Process (Shared Memory con processi separati)
        # Metodo spostato in orchestrator.py
        pr_pres, pr_ass, err_total_pc = orch.query_shared_memory(test_presenti, test_assenti, n_process=w)
        print(f"⚠️ ERRORI(processi): attesi {len(test_presenti)}, ottenuti {err_total_pc}")

        risultati_process[w] = (pr_pres, pr_ass)

    print("  -> Esecuzione completata!                        \n")

    # --- RIEPILOGO RISULTATI ---
    print("=" * 90)
    print(f"{'RIEPILOGO THROUGHPUT (Query al secondo)':^90}")
    print("=" * 90)
    print(f"{'Baseline Sequenziale':<25} | Presenti: {seq_pres:,.0f} q/s | Assenti: {seq_ass:,.0f} q/s")
    print("-" * 90)

    print(
        f"{'Worker':<6} | {'Thread (Presenti)':<20} | {'Thread (Assenti)':<20} | {'Process (Presenti)':<20} | {'Process (Assenti)':<20}")
    print("-" * 90)

    for w in worker_counts:
        th_p, th_a = risultati_thread[w]
        pr_p, pr_a = risultati_process[w]

        str_th_p = f"{th_p:<20,.0f}" if th_p is not None else f"{'N/A (GIL)':<20}"
        str_th_a = f"{th_a:<20,.0f}" if th_a is not None else f"{'N/A (GIL)':<20}"

        print(f"{w:<6} | {str_th_p} | {str_th_a} | {pr_p:<20,.0f} | {pr_a:<20,.0f}")
    print("=" * 90)

    # Analisi qualitativa dei Falsi Positivi
    analyze_false_positives_similarity(orch.bloom, em, dataset_training, test_assenti)





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
    main()
