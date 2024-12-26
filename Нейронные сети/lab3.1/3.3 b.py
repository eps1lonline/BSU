# Словарь с парами слов (слово: синоним)
synonyms_dict = {
    "happy": "joyful",
    "joyful": "happy",
    "big": "large",
    "large": "big",
    "fast": "quick",
    "quick": "fast",
    "good": "great",
    "great": "good",
    "hot": "warm",
    "warm": "hot"
}

# Слово, для которого нужно найти синоним
word = "warm"

if word in synonyms_dict:
    synonym = synonyms_dict[word]
    print(f"Синоним для слова '{word}' - '{synonym}'")
else:
    print(f"Для слова '{word}' нет синонима в данном словаре")