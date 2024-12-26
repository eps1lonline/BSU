def capitalize(word):
    return chr(ord(word[0]) - 32) + word[1:] if word else ''

def capitalize_sentence(sentence):
    words = sentence.split()
    capitalized_words = [capitalize(word) for word in words]
    return ' '.join(capitalized_words)

sentence = input("Enter a sentence: ")
print(capitalize_sentence(sentence))