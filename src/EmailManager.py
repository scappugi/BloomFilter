import unicodedata
from faker import Faker
import random

fake = Faker('it_IT')
class EmailManager:
#     Questo modulo si occupa di normalizzare le email e di generare email fittizie per i test.
    def normalize_email(self,email):
        # Tutto in minuscolo
        email = email.lower().strip()

        # Rimuove gli accenti
        email = unicodedata.normalize('NFKD', email).encode('ASCII', 'ignore').decode('ASCII')

        return email


    def generate_complex_email(self, q=1):
        emails = []

        for _ in range(q):
            name = fake.first_name().lower()
            last_name = fake.last_name().lower()

            suffixes = [
                str(random.randint(1, 99)),
                str(random.randint(1970, 2010)),
                "admin", "ufficio", "recapito",
                fake.word()
            ]

            detail = random.choice(suffixes)
            domain = fake.free_email_domain()
            # se è un numero non metto il punto, altrimenti sì
            suffix_sep = "" if detail.isdigit() else "."
            # costruisco l'email con un formato più complesso, ad esempio:
            emails.append(f"{name}.{last_name}{suffix_sep}{detail}@{domain}")

        return emails

    def generate_simple_email(self, q=1):
        emails = []

        for _ in range(q):
            name = fake.first_name().lower()
            last_name = fake.last_name().lower()

            domain = fake.free_email_domain()

            emails.append(f"{name}.{last_name}@{domain}")

        return emails

