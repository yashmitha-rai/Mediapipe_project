import tkinter as tk
import re
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import text

# -----------------------------
# LOAD SENTIMENT CLASSIFIER
# -----------------------------
classifier_model_path = "bert_classifier.tflite"
base_options_classifier = python.BaseOptions(
    model_asset_path=classifier_model_path
)

classifier_options = text.TextClassifierOptions(
    base_options=base_options_classifier,
    max_results=1
)

classifier = text.TextClassifier.create_from_options(
    classifier_options
)

# -----------------------------
# LOAD TEXT EMBEDDER
# -----------------------------
embedder_model_path = "universal_sentence_encoder.tflite"
base_options_embedder = python.BaseOptions(
    model_asset_path=embedder_model_path
)

embedder_options = text.TextEmbedderOptions(
    base_options=base_options_embedder
)

embedder = text.TextEmbedder.create_from_options(
    embedder_options
)

# -----------------------------
# VALIDATION FUNCTION
# -----------------------------
def is_valid_sentence(text_input):
    words = text_input.split()

    if len(words) < 3:
        return False

    for word in words:
        if not re.match(r"^[A-Za-z.,!?']+$", word):
            return False

    vowel_count = sum(1 for char in text_input.lower() if char in "aeiou")
    if vowel_count < 3:
        return False

    return True


# -----------------------------
# SENTIMENT FUNCTION
# -----------------------------
def classify_text():
    text1 = text_entry1.get("1.0", tk.END).strip()
    text2 = text_entry2.get("1.0", tk.END).strip()

    if text1 == "" or text2 == "":
        result_label.config(text="Enter both sentences", fg="black")
        return

    if not is_valid_sentence(text1) or not is_valid_sentence(text2):
        result_label.config(text="Invalid sentence format", fg="orange")
        return

    # --- Sentiment Sentence 1 ---
    result1 = classifier.classify(text1)
    category1 = result1.classifications[0].categories[0]
    label1 = category1.category_name.lower()
    score1 = round(category1.score * 100, 2)

    # --- Sentiment Sentence 2 ---
    result2 = classifier.classify(text2)
    category2 = result2.classifications[0].categories[0]
    label2 = category2.category_name.lower()
    score2 = round(category2.score * 100, 2)

    sentiment_text = (
        f"Sentence 1 → {label1.capitalize()} ({score1}%)\n"
        f"Sentence 2 → {label2.capitalize()} ({score2}%)"
    )

    result_label.config(text=sentiment_text, fg="green")


# -----------------------------
# SIMILARITY FUNCTION
# -----------------------------
def check_similarity():
    text1 = text_entry1.get("1.0", tk.END).strip()
    text2 = text_entry2.get("1.0", tk.END).strip()

    if text1 == "" or text2 == "":
        similarity_label.config(text="Enter both sentences", fg="black")
        return

    # Generate embeddings
    result1 = embedder.embed(text1)
    result2 = embedder.embed(text2)

    vec1 = np.array(result1.embeddings[0].embedding)
    vec2 = np.array(result2.embeddings[0].embedding)

    cosine_similarity = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

    similarity_percent = round(float(cosine_similarity) * 100, 2)

    # Threshold logic
    if similarity_percent >= 85:
        similarity_label.config(
            text=f"Similarity: {similarity_percent}% → Highly Similar ✅",
            fg="blue"
        )
    else:
        similarity_label.config(
            text=f"Similarity: {similarity_percent}% → Not Highly Similar ❌",
            fg="red"
        )


# -----------------------------
# GUI SETUP
# -----------------------------
root = tk.Tk()
root.title("Sentiment + Similarity Analyzer")
root.geometry("700x500")

tk.Label(root, text="Sentence 1", font=("Arial", 14)).pack(pady=5)
text_entry1 = tk.Text(root, height=4, width=70)
text_entry1.pack(pady=5)

tk.Label(root, text="Sentence 2", font=("Arial", 14)).pack(pady=5)
text_entry2 = tk.Text(root, height=4, width=70)
text_entry2.pack(pady=5)

tk.Button(root, text="Analyze Sentiment",
          command=classify_text,
          font=("Arial", 12)).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 13))
result_label.pack(pady=10)

tk.Button(root, text="Check Similarity (Embedding)",
          command=check_similarity,
          font=("Arial", 12)).pack(pady=5)

similarity_label = tk.Label(root, text="", font=("Arial", 13))
similarity_label.pack(pady=15)

root.mainloop()
