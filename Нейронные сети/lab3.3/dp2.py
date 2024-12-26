def remove_words_from_file(file_name, words_file):
    try:
        # Read the list of words to remove
        with open(words_file, 'r', encoding='utf-8') as wf:
            words_to_remove = set(wf.read().split())

        # Read the content of the main file
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split the content into words and filter out the words to remove
        filtered_content = ' '.join(
            word for word in content.split() if word not in words_to_remove
        )

        # Print the filtered content
        print(filtered_content)

    except FileNotFoundError:
        print("One or both files were not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_name = '1.txt'
words_file = '2.txt'
remove_words_from_file(file_name, words_file)