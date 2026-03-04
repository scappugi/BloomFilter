import BloomFilter
import EmailManager


class EmailFilterService:
    def __init__(self, n_total, prob):
        self.bloom = BloomFilter.BloomFilter.from_probability(n_total, prob)
        self.em = EmailManager.EmailManager()

    def add_email(self, raw_email):
        clean_email = self.em.normalize_email(raw_email)
        self.bloom.add(clean_email)

    def is_email_present(self, raw_email):
        clean_email = self.em.normalize_email(raw_email)
        return self.bloom.contains(clean_email)

    def get_raw_bloom_filter(self):
        return self.bloom