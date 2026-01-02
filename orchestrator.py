import BloomFilter
import multiprocessing
import worker
from joblib import Parallel, delayed
from multiprocessing import shared_memory
import numpy as np

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
            buffer = np.zeros(m, dtype=np.uint8)
            # reduce step: unisce i risultati di tutti i worker
            for ba in results: # si blocca in attesa dei risultati se ci sono worker attivi
                arr_view = np.frombuffer(ba, dtype=np.uint8)
                np.bitwise_or(buffer, arr_view, out=buffer)

            self.bloom.bit_array = buffer.tolist()
        return self.bloom


    def split_data(self, raw_datasets, num_factors=4):

        total_items = len(raw_datasets)

        chunk_factor = num_factors # Numero di chunk per worker
        num_chunks = self.num_workers * chunk_factor
        chunk_size = max(1, total_items // num_chunks)  # Assicuriamoci che sia almeno 1
        # Suddivisione
        chunks = [raw_datasets[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
        return chunks

    def run_worker(self, chunks):
        shared_bit_array = multiprocessing.Array('b', self.bloom.m , lock = False)
        args = [(chunk, self.bloom.m, self.bloom.k) for chunk in chunks]
        with multiprocessing.Pool(initializer=worker.init_worker_shared,
                                  initargs=(shared_bit_array,),
                                  processes=self.num_workers) as pool:
            pool.map(worker.process_chunk_shared, args)

        self.bloom.bit_array = list(shared_bit_array)
        return self.bloom

    def run_joblib_worker(self, raw_datasets, num_factors=4):
        total_items = len(raw_datasets)
        num_chunks = self.num_workers * num_factors
        chunk_size = max(1, total_items // num_chunks)
        chunks = [raw_datasets[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()
        args = [(chunk, m, k) for chunk in chunks]
        #passo di map
        results = Parallel(n_jobs=self.num_workers)(
            delayed(worker.process_joblib_standard)(chunk, m, k) for chunk in chunks
        )
        buffer = np.zeros(m, dtype=np.uint8)

        #passo di reduce
        for ba in results:
            np.bitwise_or(buffer, np.frombuffer(ba,np.uint8), out=buffer)

        self.bloom.bit_array = buffer.tolist()
        return self.bloom

    def run_joblib_shared_worker(self, raw_datasets, num_factors=4):
        total_items = len(raw_datasets)
        num_chunks = self.num_workers * num_factors
        chunk_size = max(1, total_items // num_chunks)
        chunks = [raw_datasets[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

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
            self.bloom.bit_array = bit_array.tolist()

        finally:
            shm.close()
            shm.unlink()
        return self.bloom





