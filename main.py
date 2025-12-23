import csv

import BloomFilter
import EmailManager

emailManager = EmailManager.EmailManager()
with open("emails_normalizzate.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["email_normalizzata"])  # header

    for email in emailManager.generate_complex_email(500000):
        normalized = emailManager.normalize_email(email)
        writer.writerow([normalized])

print ("Bloom Filter Example")
bloom = BloomFilter.BloomFilter.from_probability(n=500000, p=0.01)

with open("emails_normalizzate.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # salta l'header

    for row in reader:
        email = row[0]  # la prima colonna è l'email normalizzata
        bloom.add(email)


# Check for existence
emails_test = emailManager.generate_complex_email(100000)
emails_test_normalized = [emailManager.normalize_email(email) for email in emails_test]
for email in emails_test_normalized:
    result = bloom.contains(email)
    print(f"{email}: {'Possibly in set' if result else ''}")


# Print Bloom filter parameters
print(f"Size (m): {bloom.get_size()}")
print(f"Hash functions (k): {bloom.get_hash_count()}")
print(f"False positive rate (p): {bloom.get_false_positive_rate()}")