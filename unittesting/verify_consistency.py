import multiprocessing
import unittest
from src import BloomFilter, orchestrator, EmailManager


class TestBloomConsistency(unittest.TestCase):
    def setUp(self):
        self.n_emails = 5000  # Abbastanza per generare collisioni, non troppo lento
        self.probability = 0.01
        self.num_workers = multiprocessing.cpu_count()
        em = EmailManager.EmailManager()
        self.dataset = em.generate_complex_email(self.n_emails)

        self.bf_seq = BloomFilter.BloomFilter.from_probability(self.n_emails, self.probability)
        for email in self.dataset:
            n_email = em.normalize_email(email)
            self.bf_seq.add(n_email)

        self.bit_array = self.bf_seq.get_bit_array()
        self.orch = orchestrator.BloomOrchestrator(self.n_emails, self.probability, self.num_workers)

    def test_joblib_shared_consistency(self):
        print(f"\n\nTesting Joblib NumPy (Workers: {self.num_workers})...")

        bf_parallel = self.orch.run_joblib_shared_worker(self.dataset,)
        self.assertEqual(self.bf_seq.m, bf_parallel.m, "Le dimensioni (m) non coincidono")
        self.assertEqual(self.bf_seq.k, bf_parallel.k, "Le dimensioni (k) non coincidono")
        self.assertEqual(self.bit_array, bf_parallel.get_bit_array(), "Le dimensioni (m) non coincidono")
        print("Joblib NumPy: Test Superato (Identico al Sequenziale)")

    def test_joblib_consistency(self):
        print(f"\n\nTesting Joblib (Workers: {self.num_workers})...")

        bf_parallel = self.orch.run_joblib_worker(self.dataset,)
        self.assertEqual(self.bf_seq.m, bf_parallel.m, "Le dimensioni (m) non coincidono")
        self.assertEqual(self.bf_seq.k, bf_parallel.k, "Le dimensioni (k) non coincidono")
        self.assertEqual(self.bit_array, bf_parallel.get_bit_array(), "Le dimensioni (m) non coincidono")
        print("Joblib: Test Superato (Identico al Sequenziale)")


    def test_multiprocessing_shared_consistency(self):
        print(f"\n\nTesting Multiprocessing shared (Workers: {self.num_workers})...")

        chunk = self.orch.split_data(self.dataset)
        bf_parallel = self.orch.run_worker(chunk)
        self.assertEqual(self.bf_seq.m, bf_parallel.m, "Le dimensioni (m) non coincidono")
        self.assertEqual(self.bf_seq.k, bf_parallel.k, "Le dimensioni (k) non coincidono")
        self.assertEqual(self.bit_array, bf_parallel.get_bit_array(), "Le dimensioni (m) non coincidono")
        print("MP: Test Superato (Identico al Sequenziale)")

    def test_multiprocessing_consistency(self):
        print(f"\n\nTesting Multiprocessing  (Workers: {self.num_workers})...")

        bf_parallel = self.orch.process_chunks(self.dataset, )
        self.assertEqual(self.bf_seq.m, bf_parallel.m, "Le dimensioni (m) non coincidono")
        self.assertEqual(self.bf_seq.k, bf_parallel.k, "Le dimensioni (k) non coincidono")
        self.assertEqual(self.bit_array, bf_parallel.get_bit_array(), "Le dimensioni (m) non coincidono")
        print("Map Reduce MP: Test Superato (Identico al Sequenziale)")

if __name__ == '__main__':
    unittest.main()
