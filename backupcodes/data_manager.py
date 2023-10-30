import os
import pickle
from keras.models import load_model
import random
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

CHAT_HISTORY = "chat_history"

def load_model_from_file(model_name="model.h5", model_dir="models"):
    model_path = os.path.join(model_dir, model_name)  # Construct the path with the directory.
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception as e:  # It's a good practice to log the exception.
            print(f"Failed to load model, error: {e}")
    else:
        print("Model file does not exist.")
    return None


def save_model_to_file(model, model_name="model.h5", model_dir="models"):
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)  # Create directory if it doesn't exist
        except Exception as e:
            print(f"Failed to create directory for models, error: {e}")
            return

    model_path = os.path.join(model_dir, model_name)
    try:
        model.save(model_path)
        print(f"Model saved at {model_path}")
    except Exception as e:
        print(f"Failed to save model, error: {e}")


def load_tokenizer(tokenizer_dir="tokenizers", tokenizer_filename="tokenizer.pickle"):
    # Create the path to the tokenizer based on the specified directory and filename
    tokenizer_path = os.path.join(tokenizer_dir, tokenizer_filename)

    # Check if the tokenizer file exists at the specified path
    if os.path.exists(tokenizer_path):
        try:
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            return tokenizer
        except Exception as e:
            print(f"Failed to load tokenizer, error: {e}")
            return None
    else:
        print("Tokenizer file does not exist.")
        return None
    

def save_tokenizer(tokenizer, tokenizer_dir="tokenizers", tokenizer_filename="tokenizer.pickle"):
    # Check if the directory exists, if not, create it.
    if not os.path.exists(tokenizer_dir):
        try:
            os.makedirs(tokenizer_dir)
        except Exception as e:
            print(f"Failed to create directory for tokenizer, error: {e}")
            return

    # Create the full path with the directory and filename
    tokenizer_path = os.path.join(tokenizer_dir, tokenizer_filename)

    try:
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved at {tokenizer_path}")
    except Exception as e:
        print(f"Failed to save tokenizer, error: {e}")


def save_chat_history(message, user_id, directory="chat_history"):
    # Ensure the message ends with a period. If it already does, avoid adding an extra one.
    if not message.strip().endswith('.'):
        message = message.strip() + '.'  # Add a period at the end if one is not present.

    # Check if the directory exists and if not, create it.
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the filename; each user should have their unique file.
    # This assumes that 'user_id' uniquely identifies your user.
    filename = os.path.join(directory, f"{user_id}_chat_history.txt")

    # Append the message to the file in the designated directory.
    with open(filename, "a") as file:  # 'a' for append mode
        file.write(message + " ") 


def load_chat_history(user_id, directory="chat_history"):
    filename = os.path.join(directory, f"{user_id}_chat_history.txt")
    if os.path.exists(filename):
        with open(filename, "r") as file:
            chat_history = file.read()
            messages = chat_history.split('. ')
            return messages, chat_history
    else:
        print(f"No chat history found for user_id {user_id}")
        return "" , "" # return an empty string if no history is found
    


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))  # Avoid punctuations
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # We have replaced enough words
            break

    # If no word was replaced, we return the original list of words
    if num_replaced == 0:
        return words

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').replace('-', ' ')
            synonyms.add(synonym) 
    return synonyms

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0    
    while len(synonyms) < 1:
        random_word = random.choice(new_words)
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(list(synonyms))
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

def random_deletion(words, n):
    if len(words) == 1:
        return words

    new_words = words.copy()
    for _ in range(n):
        random_idx = random.randint(0, len(new_words)-1)
        del new_words[random_idx]

    # If all words are deleted, return a random word
    if len(new_words) == 0:
        random_idx = random.randint(0, len(words)-1)
        return [words[random_idx]]

    return new_words

def get_synonyms(word, pos=None):
    """
    Get synonyms of the word, optionally considering its part of speech.

    :param word: str, the word to find synonyms for.
    :param pos: str, part of speech of the word.
    :return: list of synonyms.
    """
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            synonym = synonym.replace('-', ' ')
            if synonym.lower() != word.lower():  # ignore the word itself
                synonyms.add(synonym)
    return list(synonyms)

def replace_synonyms(sentence):
    """
    Replace words in the sentence with synonyms.

    :param sentence: str, the sentence to replace words in.
    :return: str, the sentence with words replaced.
    """
    # Tokenize and get part of speech
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    # Map NLTK part of speech to WordNet part of speech
    pos_map = {'NN': wordnet.NOUN, 'JJ': wordnet.ADJ, 'VB': wordnet.VERB, 'RB': wordnet.ADV}

    replaced = []
    for word, pos in tagged:
        wordnet_pos = pos_map.get(pos[:2])
        synonyms = get_synonyms(word, pos=wordnet_pos) if wordnet_pos else None

        # If we have synonyms, pick a random one, else keep the original word
        if synonyms:
            replacement = random.choice(synonyms)
            replaced.append(replacement)
        else:
            replaced.append(word)

    # Reconstruct the sentence
    new_sentence = ' '.join(replaced)
    return new_sentence


def augment_chat_messages(chat_messages, num_augmented_per_message=5):
    """
    Augment each chat message and keep both the original and augmented versions.

    :param chat_messages: List of original chat messages.
    :param num_augmented_per_message: Number of augmented versions to create per message.
    :return: New list containing both original and augmented messages.
    """
    all_messages = []

    for message in chat_messages:
        # Always include the original message
        all_messages.append(message)

        words = message.split(' ')
        print(words)
        for _ in range(num_augmented_per_message):
            # Apply different augmentation techniques
            augmented_message = ' '.join(random.choice([
                synonym_replacement(words, 2)
            ]))
            all_messages.append(augmented_message)

    return all_messages


def prepare_history_data_for_training(user_id, directory="chat_history"):
    """
    Load the chat history for a given user, augment the data, and prepare it for training.

    :param user_id: The user's unique identifier.
    :param directory: The directory where chat histories are stored.
    :return: A string that combines original and augmented chat messages, ready for training.
    """

    messages, chat_history = load_chat_history(user_id)
    augmented_messages = augment_chat_history_with_synonyms(messages)


    qa_pair = load_base_data()
    data = append_qa_to_augmented_text(augmented_messages,qa_pair)


    # Step 3: Combine all messages into a single text, separating them by a period
    
    return data, chat_history


def augment_chat_history_with_synonyms(chat_history):
    """
    Augment the chat history by adding new sentences with synonyms alongside the original sentences.

    :param chat_history: str, the original chat history.
    :return: str, the augmented chat history with original and new sentences.
    """
    # Split the chat history into individual sentences. Here we use the full stop as a separator.
    augmented_sentences = []
     
    for sentence in chat_history:
        sentence = sentence.strip()  
        # print(sentence)# Remove leading/trailing white spaces
        if sentence:  # Ensure the sentence is not empty
            try:
                # Replace words with synonyms, where possible, and add to new chat history.
                augmented_sentence = replace_synonyms(sentence)
                augmented_sentences.append(sentence)  # Original sentence
                augmented_sentences.append(augmented_sentence) 
                augmented_sentences.append(sentence)
                print(augmented_sentences) # New sentence with synonyms
            except Exception as e:
                print(f"Failed to augment a sentence with synonyms: '{sentence}'. Error: {str(e)}")
                # If an error occurs, we add the original sentence.
                augmented_sentences.append(sentence)

    # Combine augmented sentences into a new chat history text.
    # Here, we join the sentences with '. ' to reconstruct the chat history with individual sentences separated by a full stop.
    # Additionally, we add a space after each sentence for better readability and structure.
    augmented_text = '. '.join(augmented_sentences) + '. ' if augmented_sentences else ''
 
    return augmented_text

def append_base_text(augmented_text, base_text):
    """
    Append an amount of the base text to the augmented text.

    :param augmented_text: str, the augmented text from chat history.
    :param base_text: str, the original base text.
    :return: str, the combined text.
    """
    # Calculate the size of 10% of the augmented text
    size_to_append = len(augmented_text) // 10

    # Check if the base text is smaller than the size we want to append
    if len(base_text) < size_to_append:
        text_to_append = base_text
    else:
        # If it's larger, then we take the first 'size_to_append' characters from the base text
        text_to_append = base_text[:size_to_append]

    # Combine the augmented text and the portion of the base text
    combined_text = augmented_text + " " + text_to_append  # Add a space between for separation

    return combined_text

def load_base_data():
    """
    Load question and answer data from a spreadsheet file.

    :param file_path: str, path to the spreadsheet file containing the QA pairs.
    :return: list of tuples, each tuple is a pair of (question, answer).
    """
    file_path = os.path.join(CHAT_HISTORY, 'conversation.csv')
    # Load the spreadsheet file
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Invalid file format. Please use '.csv' or '.xlsx' file formats.")

    # Check if necessary columns exist
    if 'question' not in data.columns or 'answer' not in data.columns:
        raise ValueError("Missing required columns ('question', 'answer') in the file.")

    # Extract question and answer pairs
    qa_pairs = list(zip(data['question'], data['answer']))

    return qa_pairs


def convert_qa_to_text(qa_pairs):
    """
    Convert question-answer pairs into a text format.

    :param qa_pairs: list of tuples, each tuple is a pair of (question, answer).
    :return: str, concatenated questions and answers.
    """
    qa_text = ''
    for question, answer in qa_pairs:
        qa_text += question + ' ' + answer + ' '  # Adding a space between QA pairs for clarity
    return qa_text

def get_random_sentences(full_text, number_of_sentences):
    """
    Get random whole sentences from a text.

    :param full_text: str, the entire text to select the sentences from.
    :param number_of_sentences: int, the number of sentences to randomly extract.
    :return: str, the concatenated random sentences.
    """
    # Split the text into sentences
    sentences = sent_tokenize(full_text)

    # If the total number of sentences is less than the number we want to select,
    # just return all sentences
    if len(sentences) <= number_of_sentences:
        return ' '.join(sentences)

    # Randomly select sentences
    selected_sentences = random.sample(sentences, number_of_sentences)
    return ' '.join(selected_sentences)

def append_qa_to_augmented_text(augmented_text, qa_pairs, percentage=10):
    """
    Append a percentage of question-answer text to the augmented text based on the number of sentences.

    :param augmented_text: str, the augmented text.
    :param qa_pairs: list of tuples, each tuple contains (question, answer).
    :param percentage: int, the percentage of the QA text to take relative to the number of sentences.
    :return: str, the combined text.
    """
    # Convert QA pairs to text
    full_qa_text = ' '.join([' '.join(pair) for pair in qa_pairs])  # This joins all questions and answers into a single string

    # Calculate the number of sentences in the augmented text
    num_sentences_aug_text = len(sent_tokenize(augmented_text))

    # Calculate the number of sentences to select from the QA text
    num_sentences_to_select = (num_sentences_aug_text * percentage) // 100

    # Select sentences from the QA text
    selected_qa_sentences = get_random_sentences(full_qa_text, num_sentences_to_select)

    # Combine the selected sentences with the augmented text
    combined_text = augmented_text + " " + selected_qa_sentences  # Add a space between for separation

    return combined_text
