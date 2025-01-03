from collections import Counter
import re

#text = "Даy Даy Даy Даy ДаyДаyДаy Даy Даy Даy Даy Даy ДаyДаyДаyДаy Даy Даy Даy Даy текст: в первой строке задано число строк, далее идут сами строки. Выведите слово,\nкоторое в этом тексте встречается чаще всего. Если таких слов несколько, выведите то,\nкоторое меньше в лексикографическом порядке."
text = "a b c d d b"
# Разделить текст на слова, учитывая только буквы и цифры (остальные символы удаляются)
words = re.findall(r'\w+', text.lower())

# Подсчитать количество вхождений каждого слова
word_counts = Counter(words)

# Найти слово, которое встречается чаще всего и меньше всего в лексикографическом порядке
most_common_words = sorted(word_counts, key=lambda x: (-word_counts[x], x))

print("Слово или слог, который встречается чаще всего: '", most_common_words[0], "'")