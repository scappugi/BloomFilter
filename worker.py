import mmh3

import BloomFilter
import EmailManager
import csv
import multiprocessing
import os
import time

class BloomWorker:
    toShare = []
    _email_manager = EmailManager.EmailManager()

# Metodo per Mapper
    @staticmethod
    def process_chunk(args):
        raw_emails_chunk, m, k = args
        all_indices = []

        # Per ogni email nel pacchetto
        for raw_email in raw_emails_chunk:
            # Normalizza l'email
            email = BloomWorker._email_manager.normalize_email(raw_email)

            # Calcola gli indici degli hash
            for i in range(k):
                # i è il seed della funzione di hash
                index = mmh3.hash(str(email), i) % m
                all_indices.append(index)

        return all_indices

    @staticmethod
    def process_chunk_shared(args):
        raw_emails_chunk, m, k = args
        # Per ogni email nel pacchetto
        for raw_email in raw_emails_chunk:
            email = BloomWorker._email_manager.normalize_email(raw_email)

            for i in range(k):
                index = mmh3.hash(str(email), i) % m

                # Aggiorna l'array di bit condiviso
                BloomWorker.toShare[index] = 1

    @staticmethod
    def init_worker_shared(shared_array):
        BloomWorker.toShare = shared_array