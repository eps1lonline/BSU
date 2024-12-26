text = "Hello world! Its a me, Mario. Hello world!"
words = text.split()
word_count = {}

for i, word in enumerate(words):
    count = sum(1 for w in words[:i] if w == word)
    word_count[word] = count

# Вывод результатов
for word, count in word_count.items():
    print(f'{word}: {count}')