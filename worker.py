import mmh3

import BloomFilter
import EmailManager
from bitarray import bitarray
import numpy as np
from multiprocessing import shared_memory
# Variabili globale a livello di modulo per il worker corrente
toShare = None
_email_manager = EmailManager.EmailManager()


def init_worker_shared(shared_array):
    #Inviamo il puntatore una sola volta all'avvio del processo worker. Il worker lo salva in una variabile globale e lo usa per sempre.
    global toShare
    toShare = shared_array

# Metodo per Mapper
def process_chunk(args):
    raw_emails_chunk, m, k = args
    all_indices = bitarray(m)
    all_indices.setall(0)

    # Per ogni email nel pacchetto
    for raw_email in raw_emails_chunk:
        # Normalizza l'email
        email = _email_manager.normalize_email(raw_email)
        indices = BloomFilter.BloomFilter.calculate_hashes(email, m, k)
        # Aggiorna l'array di bit locale
        for index in indices:
            all_indices[index] = 1

    # Ritorna l'array di bit locale (seriealizzazione piu veloce di una classica lista)
    return all_indices


def process_chunk_shared(args):
    raw_emails_chunk, m, k = args
    local_array = toShare #ottiene il riferimento all'array di bit condiviso
    # Per ogni email nel pacchetto
    for raw_email in raw_emails_chunk:
        email = _email_manager.normalize_email(raw_email)
        indices = BloomFilter.BloomFilter.calculate_hashes(email, m, k)
        # Aggiorna l'array di bit condiviso
        for index in indices:
            toShare[index] = 1


def process_joblib_standard(chunk, m, k):
    local_bits = bitarray(m)
    calc_hashes = BloomFilter.BloomFilter.calculate_hashes

    for raw_email in chunk:
        email = _email_manager.normalize_email(raw_email)
        for idx in calc_hashes(email, m, k):
            local_bits[idx] = 1

    return local_bits

def process_joblib_shared(raw_emails_chunk, m, k, shm_name, dtype):
    shm = shared_memory.SharedMemory(name = shm_name)
    shared_array = np.ndarray(shape=(m,), dtype=dtype, buffer=shm.buf)
    for raw_email in raw_emails_chunk:
        email = _email_manager.normalize_email(raw_email)
        indices = BloomFilter.BloomFilter.calculate_hashes(email, m, k)
        # Aggiorna l'array di bit condiviso
        for index in indices:
            shared_array[index] = 1
    #chiude il collegamento
    shm.close()

