# synthetic_data/generate_faker.py
import pandas as pd
from faker import Faker

def generate_faker_text(n_rows=1000):
    fake = Faker()
    data = {
        "text": [fake.sentence(nb_words=10) for _ in range(n_rows)],
        "label": [fake.random_element(["positive", "negative", "neutral"]) for _ in range(n_rows)]
    }
    return pd.DataFrame(data)
