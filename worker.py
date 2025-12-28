import mmh3

import BloomFilter
import EmailManager
import csv
import multiprocessing
import os
import time

# Variabili globale a livello di modulo per il worker corrente
toShare = []
_email_manager = EmailManager.EmailManager()


# Metodo per Mapper
def process_chunk(args):
    raw_emails_chunk, m, k = args
    all_indices = []

    # Per ogni email nel pacchetto
    for raw_email in raw_emails_chunk:
        # Normalizza l'email
        email = _email_manager.normalize_email(raw_email)
        indices = BloomFilter.BloomFilter.calculate_hashes(email, m, k)
        all_indices.extend(indices)

    return all_indices


def process_chunk_shared(args):
    raw_emails_chunk, m, k = args
    # Per ogni email nel pacchetto
    for raw_email in raw_emails_chunk:
        email = _email_manager.normalize_email(raw_email)
        indices = BloomFilter.BloomFilter.calculate_hashes(email, m, k)
        # Aggiorna l'array di bit condiviso
        for index in indices:
            toShare[index] = 1


def init_worker_shared(shared_array):
    global toShare
    toShare = shared_array
