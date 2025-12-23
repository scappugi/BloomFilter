import hashlib
import math

import mmh3 as mmh3


class BloomFilter:
    def __init__(self, n: int, m: int, k: int, p: float):
        self.n = n
        self.m = m
        self.k = k
        self.p = p
        self.bit_array = [0] * m

    @classmethod
    def from_probability(cls, n: int, p: float):
        m = int(-(n * math.log(p)) / (math.log(2) ** 2))
        k = max(1, round((m / n) * math.log(2)))
        return cls(n, m, k, p)

    @classmethod
    def from_memory(cls, n: int, m: int):
        k = max(1, round((m / n) * math.log(2)))
        p = (1 - math.exp(-k * n / m)) ** k
        return cls(n, m, k, p)

    @classmethod
    def from_number_of_hashes(cls, n: int, k: int):
        m = max(1, round((n * k) / math.log(2)))
        p = (1 - math.exp(-k * n / m)) ** k
        return cls(n, m, k, p)

    def _hashes(self, item):
        indices = []
        for i in range(self.k):
            indices.append(mmh3.hash(str(item), i) % self.m)
        return indices

    def add(self, item):
        for index  in self._hashes(item):
            self.bit_array[index] = 1

    def contains(self, item) -> bool:
        for index in self._hashes(item):
            if self.bit_array[index] == 0:
                return False
        return True

    def get_size(self) -> int:
        return self.m

    def get_hash_count(self) -> int:
        return self.k

    def get_false_positive_rate(self) -> float:
        return self.p

    # Metodo per Reducer
    def add_indices(self, indices):
        for index in indices:
            self.bit_array[index] = 1



