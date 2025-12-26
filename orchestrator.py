import BloomFilter
import multiprocessing
import worker


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
        print("Processing chunks...")

        total_items = len(raw_datasets)

        chunk_factor = num_factors # Numero di chunk per worker
        num_chunks = self.num_workers * chunk_factor
        chunk_size = max(1, total_items // num_chunks)  # Assicuriamoci che sia almeno 1
        # Suddivisione
        chunks = [raw_datasets[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

        # Parametri del Bloom Filter
        m = self.bloom.get_size()
        k = self.bloom.get_hash_count()
        args = [(chunk, m, k) for chunk in chunks]

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            # map step: distribuisce i chunk ai worker
            results = pool.imap_unordered(worker.BloomWorker.process_chunk, args)

            # reduce step: unisce i risultati di tutti i worker
            processed_chunks = 0
            for indices in results:
                self.bloom.add_indices(indices)
                processed_chunks += 1
                print(f"Processed chunks: {processed_chunks}/{len(chunks)}", end="\r")

        print("\nAll chunks processed.")
        return self.bloom





