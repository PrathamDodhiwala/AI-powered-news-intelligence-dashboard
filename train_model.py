import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

technology = [
    "Artificial Intelligence transforms industries",
    "New machine learning model released",
    "Cloud computing adoption grows rapidly",
    "Cybersecurity threats increase worldwide",
    "Quantum computing breakthrough announced",
]

finance = [
    "Stock market closes higher today",
    "Investors remain optimistic about economy",
    "Interest rates remain unchanged",
    "Banks report strong quarterly profits",
    "Financial markets experience volatility",
]

health = [
    "New vaccine research shows promise",
    "Healthcare technology continues improving",
    "Doctors discover treatment breakthrough",
    "Medical innovations reduce patient risk",
    "Health organizations publish new study",
]

fake_news = [
    "Aliens control world governments secretly",
    "Drinking oil cures every disease instantly",
    "Moon made entirely of gold scientists say",
    "Invisible dragons found in city center",
    "Time traveler wins lottery repeatedly",
]

records = []

for _ in range(1000):

    article_type = np.random.choice(["technology", "finance", "health", "fake"])

    if article_type == "technology":
        text = np.random.choice(technology)
        label = 0

    elif article_type == "finance":
        text = np.random.choice(finance)
        label = 0

    elif article_type == "health":
        text = np.random.choice(health)
        label = 0

    else:
        text = np.random.choice(fake_news)
        label = 1

    records.append([text, label])

df = pd.DataFrame(records, columns=["Article", "FakeNews"])

df.to_csv("news_dataset.csv", index=False)

X = df["Article"]
y = df["FakeNews"]

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Training completed successfully")
