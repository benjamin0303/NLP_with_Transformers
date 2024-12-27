#from datasets import list_datasets # Depreciated in 3.2.0
from huggingface_hub import list_datasets # Use this instead
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

#all_datasets = list(list_datasets())
#print(f'There are {len(all_datasets)} datasets available on the Hub')
#print(f'The first 10 are: {all_datasets[:10]}')

emotions = load_dataset('dair-ai/emotion')
print(type(emotions))
print(emotions)
for key in emotions.keys():
    print(key)
    
train_ds = emotions['train']
print(len(train_ds))

for i in range(10):
    print(train_ds[i])
    
print(train_ds.features)

print('---Now set to pandas format---')
emotions.set_format(type='pandas')
df = emotions['train'][:]
print(df.head())

def label_int2str(row):
    return emotions['train'].features['label'].int2str(row)

df['label_name'] = df['label'].apply(label_int2str)
print(df.head())

print(emotions.shape)
print(emotions['train'].shape)

print(emotions['train'].features['label'])

print(df['label'].shape)


df['label_name'].value_counts().plot(kind='bar')
plt.title("Frequency of Classes")
plt.savefig('emotions_histogram.png')


df['Words Per Tweet'] = df['text'].str.split().apply(len)
df.boxplot('Words Per Tweet', by='label_name', grid=False, showfliers=False, color='k')
plt.suptitle('')
plt.xlabel('')
plt.savefig('emotions_boxplot.png')

emotions.reset_format()

## Tokenizing
text = 'Tokenizing text is a core task of NLP.'

# Character level tokenization
tokenized_text = list(text)
#print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

input_ids = [token2idx[ch] for ch in tokenized_text]
print(input_ids)

import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)

# Word level tokenization
tokenized_text = text.split()
print(tokenized_text)

# Subword level tokenization

from transformers import AutoTokenizer

model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

encoded_text=tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))
print(tokenizer.vocab_size)
print(tokenizer.model_max_length)
print(tokenizer.model_input_names)