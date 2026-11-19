#!/bin/bash
# Reference solution (Harbor's oracle convention).
cat > /app/solution.py <<'PY'
import string


def count_words(text):
    counts = {}
    for piece in text.split():
        word = piece.strip(string.punctuation).lower()
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts
PY
