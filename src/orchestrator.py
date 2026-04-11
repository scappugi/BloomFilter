import time

from bitarray import bitarray

from src import BloomFilter, EmailManager
import multiprocessing
from src import worker
from joblib import Parallel, delayed
from multiprocessing import shared_memory
import numpy as np
import concurrent.futures


class BloomOrchestrator:
    def __init__(self, n_total, arg, num_workers=None):
        """
        :param n_total: Numero stimato di elementi da inserire.
        :param arg: Probabilità di falso positivo (float tra 0 e 1) o numero di funzioni di hash (int >= 1).
        :param num_workers: Numero di processi paralleli (default: tutti i core disponibili).
        """
        self.n_total = n_total
        self.arg = arg
        # Se num_workers è None o 0, usa tutti i core della CPU
        self.num_workers = num_workers if num_workers else multiprocessing.cpu_count()
        # Bloom Filter (il "Reducer" finale)
        if arg < 1.0:
            self.bloom = BloomFilter.BloomFilter.from_probability(self.n_total, arg)

        elif arg >= 1 and isinstance(arg, int):
            self.bloom = BloomFilter.BloomFilter.from_number_of_hashes(self.n_total, arg)

    #@profile
    def process_chunks(self, raw_datasets, num_factors=4):

        """
        raw_datasets: Lista di email raw da processare.
        num_factors: Numero di chunk per worker.
        """

        chunks = self.split_data(raw_datasets, num_factors)
        # Parametri del Bloom Filter
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()
        args = [(chunk, m, k) for chunk in chunks]

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            # map step: distribuisce i chunk ai worker
            results = pool.imap_unordered(worker.process_chunk, args)
            #buffer = np.zeros(m, dtype=np.uint8)

            #creo il bitarray e lo azzero (per sicurezza)
            global_bitarray = bitarray(m)
            global_bitarray.setall(0)

            # reduce step: unisce i risultati di tutti i worker
            for ba in results: # si blocca in attesa dei risultati se ci sono worker attivi
                # arr_view = np.frombuffer(ba, dtype=np.uint8)
                # np.bitwise_or(buffer, arr_view, out=buffer)
                global_bitarray |= ba

            self.bloom.bit_array = global_bitarray
        return self.bloom

#questi due metodi servono per la versione shared memory
    def split_data(self, raw_datasets, num_factors=4):
        total_items = len(raw_datasets)
        chunk_factor = num_factors # Numero di chunk per worker
        num_chunks = self.num_workers * chunk_factor
        chunk_size = max(1, total_items // num_chunks)  # Assicuriamoci che sia almeno 1
        # Suddivisione
        chunks = [raw_datasets[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
        return chunks

#metodo per la gestione di un caso di memory shared
    #@profile
    def run_worker(self, raw_datasets, num_factors=4):
        chunks = self.split_data(raw_datasets, num_factors)
        shared_bit_array = multiprocessing.Array('b', self.bloom.m, lock=False)
        args = [(chunk, self.bloom.m, self.bloom.k) for chunk in chunks]
        with multiprocessing.Pool(initializer=worker.init_worker_shared,
                                  initargs=(shared_bit_array,),
                                  processes=self.num_workers) as pool:
            pool.map(worker.process_chunk_shared, args)

            arr_view = np.frombuffer(shared_bit_array, dtype=np.uint8)
            packed_bytes = np.packbits(arr_view, bitorder='big')
            final_bloom = bitarray()
            final_bloom.frombytes(packed_bytes.tobytes())

            final_bloom = final_bloom[:self.bloom.m]

            #Vecchia versione (inefficiente) utilizzata nei grafici
            #final_bloom = bitarray()
            #final_bloom.extend(shared_bit_array)

        self.bloom.bit_array = final_bloom
        return self.bloom

    def run_joblib_worker(self, raw_datasets, num_factors=4):
        chunks = self.split_data(raw_datasets, num_factors)
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()
        args = [(chunk, m, k) for chunk in chunks]
        #passo di map
        results = Parallel(n_jobs=self.num_workers)(
            delayed(worker.process_joblib_standard)(chunk, m, k) for chunk in chunks
        )
        #buffer = np.zeros(m, dtype=np.uint8)
        buffer = bitarray(m)
        buffer.setall(0)

        #passo di reduce
        for ba in results:
            #np.bitwise_or(buffer, np.frombuffer(ba,np.uint8), out=buffer)
            buffer |= ba #bitarray supporta l'operatore OR direttamente

        self.bloom.bit_array = buffer
        return self.bloom

    def run_joblib_shared_worker(self, raw_datasets, num_factors=4):
        chunks = self.split_data(raw_datasets, num_factors)

        #shared_bit_array = multiprocessing.Array('b', self.bloom.m , lock = False)

        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()
        dtype = np.int8
        size_in_bytes = m * np.dtype(dtype).itemsize
        shm = shared_memory.SharedMemory(create=True, size=size_in_bytes)

        try:
            #serve solo a creare una zona di memoria con tutti 0 (usermo bitarry solo per lettura)
            bit_array = np.ndarray(shape = (m,), dtype = dtype, buffer = shm.buf)
            bit_array.fill(0)
            Parallel(n_jobs=self.num_workers)(
                delayed(worker.process_joblib_shared)(chunk, m, k, shm.name, 'int8') for chunk in chunks
            )
            final_bloom = bitarray()
            final_bloom.extend(bit_array)
            self.bloom.bit_array = final_bloom

        finally:
            shm.close()
            shm.unlink()
        return self.bloom

    # metodo per provare a fare load balancing
    def process_dynamic(self, raw_datasets, chunk_size=1000):
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()

        # Usiamo un Manager per gestire la coda tra processi
        with multiprocessing.Manager() as manager:
            task_queue = manager.Queue()

            # Riempiamo la coda con piccoli chunk
            for i in range(0, len(raw_datasets), chunk_size):
                task_queue.put(raw_datasets[i:i + chunk_size])

            # Aggiungiamo i 'sentinel' per fermare i worker
            for _ in range(self.num_workers):
                task_queue.put(None)

            with multiprocessing.Pool(processes=self.num_workers) as pool:
                # funzione worker che legge dalla coda
                results = pool.map(worker.process_from_queue, [(task_queue, m, k)] * self.num_workers)

                # Classico reduce
                global_bitarray = bitarray(m)
                global_bitarray.setall(0)
                for ba in results:
                    global_bitarray |= ba

                self.bloom.bit_array = global_bitarray
        return self.bloom

    def process_dynamic_imap(self, raw_datasets, chunk_size=5000):
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()

        # Prepariamo gli argomenti (uno per ogni mini-chunk)
        chunks = [raw_datasets[i:i + chunk_size] for i in range(0, len(raw_datasets), chunk_size)]
        args = [(c, m, k) for c in chunks]

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            # imap_unordered distribuisce i chunk ai worker non appena sono liberi
            results = pool.imap_unordered(worker.process_chunk, args)

            global_bitarray = bitarray(m)
            global_bitarray.setall(0)
            for ba in results:
                global_bitarray |= ba

        return self.bloom

    ###############################################################################################################
    #                                           Metodi per noGIL
    ###############################################################################################################

    def run_threaded_worker_bytearray(self, raw_datasets, num_factors=4):

        chunks = self.split_data(raw_datasets, num_factors)
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()
        args = [(chunk, m, k) for chunk in chunks]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(worker.process_thread_bytearray, args))
            final_array = np.zeros(self.bloom.m, dtype=np.uint8)

            for ba in results:
                arr_view = np.frombuffer(ba, dtype=np.uint8)
                np.bitwise_or(final_array, arr_view, out=final_array)

            # Conversione finale e assegnazione
            #self.bloom.bit_array = bitarray(final_array.tolist()) vecchia assegnazione (inefficiente)

            packed_bytes = np.packbits(final_array, bitorder='big')

            final_bloom_bits = bitarray()
            final_bloom_bits.frombytes(packed_bytes.tobytes())

            self.bloom.bit_array = final_bloom_bits[:m]

        return self.bloom

    def run_threaded_worker(self, raw_datasets, num_factors=4):

        chunks = self.split_data(raw_datasets, num_factors)
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()
        args = [(chunk, m, k) for chunk in chunks]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(worker.process_thread, args))
            final_bitarray = bitarray(self.bloom.m)
            final_bitarray.setall(0)
            for ba in results:
                final_bitarray |= ba

            self.bloom.bit_array = final_bitarray
        return self.bloom

    def run_threaded_shared(self, raw_datasets, num_factors=4):
        """
        Esegue il Bloom Filter in ambiente Python 3.14 (Free-Threaded)
        utilizzando un array NumPy condiviso tra i thread.
        """
        chunks = self.split_data(raw_datasets, num_factors)
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()
        args = [(chunk, m, k) for chunk in chunks]

        shared_array = np.zeros(m, dtype=np.uint8)
        worker.init_worker_shared(shared_array)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                list(executor.map(worker.process_thread_shared, args))

            packed_bytes = np.packbits(shared_array, bitorder='big')

            final_bloom_bits = bitarray()
            final_bloom_bits.frombytes(packed_bytes.tobytes())

            self.bloom.bit_array = final_bloom_bits[:m]

        finally:
            worker.toShare = None

        return self.bloom

    #######################################################
    # Metodi per query
    #######################################################

    def query_parallel(self, test_presenti, test_assenti, n_threads=None):
        """Esegue delle query usando thread (No-GIL)."""

        n_threads = n_threads if n_threads else self.num_workers

        # Usa il metodo split_data già presente nella classe
        chunks_presenti = self.split_data(test_presenti, 1)
        chunks_assenti = self.split_data(test_assenti, 1)

        # Test Presenti
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker.worker_query, self.bloom, chunk) for chunk in chunks_presenti]
            total = sum(f.result() for f in concurrent.futures.as_completed(futures))
        tempo_presenti = time.perf_counter() - start_time

        # Test Assenti
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker.worker_query, self.bloom, chunk) for chunk in chunks_assenti]
            [f.result() for f in concurrent.futures.as_completed(futures)]
        tempo_assenti = time.perf_counter() - start_time

        thr_p = len(test_presenti) / tempo_presenti
        thr_a = len(test_assenti) / tempo_assenti
        return thr_p, thr_a, total

    def query_shared_memory(self, test_presenti, test_assenti, n_process=None):
        """Esegue query usando processi e memoria condivisa."""

        n_process = n_process if n_process else self.num_workers
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()

        # Allocazione memoria condivisa
        shm = shared_memory.SharedMemory(create=True, size=m)  # dtype uint8 è 1 byte
        try:
            shared_array = np.ndarray(shape=(m,), dtype=np.uint8, buffer=shm.buf)
            shared_array[:] = self.bloom.bit_array.tolist()

            chunks_presenti = self.split_data(test_presenti, 1)  # suddivisione semplice
            chunks_assenti = self.split_data(test_assenti, 1)

            # Esecuzione con processi
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_process) as executor:
                start_time = time.perf_counter()
                futures = [executor.submit(worker.worker_query_shared, shm.name, m, k, chunk) for chunk in
                           chunks_presenti]
                total = sum(f.result() for f in concurrent.futures.as_completed(futures))
                tempo_presenti = time.perf_counter() - start_time

            with concurrent.futures.ProcessPoolExecutor(max_workers=n_process) as executor:
                start_time = time.perf_counter()
                futures = [executor.submit(worker.worker_query_shared, shm.name, m, k, chunk) for chunk in
                           chunks_assenti]
                [f.result() for f in concurrent.futures.as_completed(futures)]
                tempo_assenti = time.perf_counter() - start_time

            thr_p = len(test_presenti) / tempo_presenti
            thr_a = len(test_assenti) / tempo_assenti
            return thr_p, thr_a, total
        finally:
            shm.close()
            shm.unlink()

    def query_sequential(self, test_presenti, test_assenti):
        """Esegue il benchmark delle query in modalità sequenziale."""
        # Recuperiamo l'istanza di EmailManager
        em = EmailManager.EmailManager()

        print("\n--- AVVIO BENCHMARK (Sequenziale) ---")

        # Test Presenti
        start_time = time.perf_counter()
        for email in test_presenti:
            self.bloom.contains(em.normalize_email(email))
        tempo_presenti = time.perf_counter() - start_time

        # Test Assenti
        start_time = time.perf_counter()
        for email in test_assenti:
            self.bloom.contains(em.normalize_email(email))
        tempo_assenti = time.perf_counter() - start_time

        thr_p = len(test_presenti) / tempo_presenti
        thr_a = len(test_assenti) / tempo_assenti

        return thr_p, thr_a

