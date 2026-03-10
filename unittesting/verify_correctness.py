import multiprocessing
import unittest
from src import BloomFilter, orchestrator, EmailManager


class TestBloomCorrectness(unittest.TestCase):
    def setUp(self):
        self.n_emails = 20000
        self.probability = 0.01
        self.num_workers = multiprocessing.cpu_count()
        self.em = EmailManager.EmailManager()
        self.dataset = self.em.generate_complex_email(self.n_emails)

        #dataset di test
        self.test_emails = []
        candidates = self.em.generate_complex_email(self.n_emails)
        for email in candidates:
            if email not in self.dataset:
                self.test_emails.append(email)

    def verify_bloom_properties(self, bf : BloomFilter.BloomFilter):
        false_negatives = 0
        false_positives = 0
        # Verifica assenza di falsi negativi (sul dataset originale)
        for email in self.dataset:
            normalized = self.em.normalize_email(email)
            if not bf.contains(normalized):
                false_negatives += 1

        self.assertEqual(false_negatives, 0,
                         f"GRAVE: Trovati {false_negatives} Falsi Negativi! Il filtro è rotto.")
        print(f"\nFalse Negatives: 0 (OK)")
        # Verifica tasso di falsi positivi (sul dataset di test)
        for email in self.test_emails:
            normalized = self.em.normalize_email(email)
            if bf.contains(normalized):
                false_positives += 1

        measured_fpr = false_positives / len(self.test_emails)
        print(f"FPR Misurato: {measured_fpr:.4f} (Target: {self.probability})")
        tolerance = 0.015
        self.assertLess(measured_fpr, self.probability + tolerance,
                        f"FPR troppo alto! Atteso ~{self.probability}, ottenuto {measured_fpr}")


    def test_sequential_bloom_correctness(self):
        print(f"\n\nTesting Sequential Bloom Filter...")
        bf_seq = BloomFilter.BloomFilter.from_probability(self.n_emails, self.probability)
        for email in self.dataset:
            n_email = self.em.normalize_email(email)
            bf_seq.add(n_email)

        self.verify_bloom_properties(bf_seq)

    def test_multiprocessing_bloom_correctness(self):
        print(f"\n\nTesting Multiprocessing Bloom Filter (Workers: {self.num_workers})...")
        orch = orchestrator.BloomOrchestrator(self.n_emails, self.probability, self.num_workers)
        bf_parallel = orch.process_chunks(self.dataset,)
        self.verify_bloom_properties(bf_parallel)

    def test_multiprocessing_shared_bloom_correctness(self):
        print(f"\n\nTesting Multiprocessing Shared Bloom Filter (Workers: {self.num_workers})...")
        orch = orchestrator.BloomOrchestrator(self.n_emails, self.probability, self.num_workers)
        chunk = orch.split_data(self.dataset)
        bf_parallel = orch.run_worker(chunk)
        self.verify_bloom_properties(bf_parallel)

    def test_joblib_bloom_correctness(self):
        print(f"\n\nTesting Joblib Bloom Filter (Workers: {self.num_workers})...")
        orch = orchestrator.BloomOrchestrator(self.n_emails, self.probability, self.num_workers)
        bf_parallel = orch.run_joblib_worker(self.dataset,)
        self.verify_bloom_properties(bf_parallel)

    def test_joblib_shared_bloom_correctness(self):
        print(f"\n\nTesting Joblib NumPy shared Bloom Filter (Workers: {self.num_workers})...")
        orch = orchestrator.BloomOrchestrator(self.n_emails, self.probability, self.num_workers)
        bf_parallel = orch.run_joblib_shared_worker(self.dataset,)
        self.verify_bloom_properties(bf_parallel)

if __name__ == '__main__':
    unittest.main()

