import pandas as pd
import re

# read csv file
df = pd.read_csv('preprocessed_validation.csv')

# define what need to be deleted
tokens_to_remove = ['startarticle', 'startparagraph', 'startsection', 'start section', 'start article', 'start paragraph']

# function to clean tokens column
def clean_tokens(tokens):
    # transform string to list
    token_list = tokens.strip('[]').replace("'", "").split(', ')
    # delete "newline" in all words that contain newline
    cleaned_tokens = [re.sub(r'newline', '', token).strip() for token in token_list if token not in tokens_to_remove]
    return cleaned_tokens

# function to delete newline
def clean_newline(text):
    # delete all "newline"
    return re.sub(r'newline', ' ', text).strip()

# clean
df['tokens'] = df['tokens'].apply(clean_tokens)
df['cleaned_text'] = df['cleaned_text'].apply(clean_newline)

# save
df.to_csv('cleaned_preprocessed_validation.csv', index=False)

print("results saved in cleaned_preprocessed_validation.csv ")


# read csv file
df = pd.read_csv('preprocessed_test.csv')

# define what need to be deleted
tokens_to_remove = ['startarticle', 'startparagraph', 'startsection', 'start section', 'start article', 'start paragraph']

# delete tokens column
def clean_tokens(tokens):
    # transform string into list
    token_list = tokens.strip('[]').replace("'", "").split(', ')
    # delete "newline" in words involving newline
    cleaned_tokens = [re.sub(r'newline', '', token).strip() for token in token_list if token not in tokens_to_remove]
    return cleaned_tokens

# function to clean newline
def clean_newline(text):
    # delete all  "newline"
    return re.sub(r'newline', ' ', text).strip()

# clean
df['tokens'] = df['tokens'].apply(clean_tokens)
df['cleaned_text'] = df['cleaned_text'].apply(clean_newline)

# save
df.to_csv('cleaned_preprocessed_test.csv', index=False)

print("results saved in cleaned_preprocessed_test.csv ")