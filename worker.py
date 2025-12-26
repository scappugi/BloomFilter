import mmh3

import BloomFilter
import EmailManager
import csv
import multiprocessing
import os
import time

class BloomWorker:
    print ("start process id:", os.getpid())
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
