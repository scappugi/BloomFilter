import csv

from src import BloomFilter, EmailManager

emailManager = EmailManager.EmailManager()
# with open("emails_normalizzate.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["email_normalizzata"])  # header
#
#     for email in emailManager.generate_complex_email(500000):
#         normalized = emailManager.normalize_email(email)
#         writer.writerow([normalized])


print("Bloom Filter Example")
bloom = BloomFilter.BloomFilter.from_probability(n=500000, p=0.01)

# Inserimento delle email nel Bloom Filter
emails_inserted_set = set()
with open("stuff/emails_normalizzate.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # salta l'header
    for row in reader:
        email = row[0]
        bloom.add(email)
        emails_inserted_set.add(email)

# Generazione test set casuale e conteggio TP/FP
test_size = 10000
true_positives = 0
false_positives = 0

# Apriamo il CSV dove salvare le TP
with open("stuff/tp_emails.csv", "w", newline="", encoding="utf-8") as tp_file:
    writer = csv.writer(tp_file)
    writer.writerow(["email"])  # header

    for _ in range(test_size):
        email = emailManager.normalize_email(emailManager.generate_complex_email(1)[0])

        if bloom.contains(email):
            if email in emails_inserted_set:
                true_positives += 1
                writer.writerow([email])  # salva la TP nel CSV
            else:
                false_positives += 1

print(f"True positives: {true_positives}")
print(f"False positives: {false_positives}")
print(f"False positive rate: {false_positives / test_size:.4f}")

# Parametri del Bloom Filter
print(f"Size (m): {bloom.get_size()}")
print(f"Hash functions (k): {bloom.get_hash_count()}")
print(f"False positive rate (p): {bloom.get_false_positive_rate()}")
