import BloomFilter
import EmailManager

emailManager = EmailManager.EmailManager()
emails = emailManager.generate_complex_email(500000)
emails_normalized = [emailManager.normalize_email(email) for email in emails]
print ("Bloom Filter Example")
# Create a Bloom filter with expected 1000 elements and 1% false positive rate
bloom = BloomFilter.BloomFilter.from_probability(n=1000000, p=0.01)

for email in emails_normalized:
    bloom.add(email)


# Check for existence
print(bloom.contains(emails_normalized[0]))  # Expected: True
print(bloom.contains(emails_normalized[4]))  # Expected: True
print(bloom.contains("simone.cappugi@virgilio.it")) # Expected: False (most likely)
emails_test = emailManager.generate_complex_email(100000)
emails_test_normalized = [emailManager.normalize_email(email) for email in emails_test]
for email in emails_test_normalized:
    result = bloom.contains(email)
    print(f"{email}: {'Possibly in set' if result else ''}")


# Print Bloom filter parameters
print(f"Size (m): {bloom.get_size()}")
print(f"Hash functions (k): {bloom.get_hash_count()}")
print(f"False positive rate (p): {bloom.get_false_positive_rate()}")