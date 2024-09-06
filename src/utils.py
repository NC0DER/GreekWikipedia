import os
import re
import nltk 
import pandas

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, concatenate_datasets
from statistics import mean
from typing import Tuple, TypeVar, List, Dict
from src.extractive import textrank_gr

# Generic type class for model and dataset objects.
Model = TypeVar('Model')
HFDataset = TypeVar('HFDataset')


def sent_tokenize(text):
    """
    Function that tokenizes texts into a list of sentences.
    """

    sentences = nltk.sent_tokenize(text, language = 'greek')
    sentences = (filter(None, re.split(r'(?<=[;;])\s+', sentence)) for sentence in sentences)
    
    return [sentence.strip() for sent_gen in sentences for sentence in sent_gen]


def word_tokenize(text):
    """
    Function that tokenizes texts into a list of words.
    """
    
    return nltk.RegexpTokenizer(r"[ ,;;.!?:-]+", gaps = True).tokenize(text)


def get_first_n_sentences(text, n):
    """
    Function that gets the first n sentences of the text.
    """

    return ' '.join(sent_tokenize(text, language = 'greek')[:n])


def reduce_text(text: str, tokenizer: Model) -> str:
    """
    Function that reduces a text, as part of 
    the reduce then summarize strategy 
    by utilizing extractive summarization.
    
    Parameters
    -----------
    text: the text to be summarized (str).
    tokenizer: huggingface tokenizer model (Model).

    Returns
    --------
    summary: the summary text (str)
    """

    # if the text is empty, return early.
    if text == '':
        return ''

    # The max supported input tokens for umt5 models is 1024.
    max_input_tokens = 1024

    # Count the number of sentences and tokens.
    sentence_count = len(list(sent_tokenize(text)))
    token_count = len(tokenizer.tokenize(text))
    tokens_per_sentence = token_count // sentence_count

    # If the text has less tokens than the max input length, return it.
    if token_count < max_input_tokens:
        summary = text
    else: # If it has more, reduce the text by using TextRank 
        # to extract its top-n sentences.
        top_n_sentences = int(max_input_tokens / tokens_per_sentence)
        summary = textrank_gr(text, n = top_n_sentences)

    return summary


def measure_token_lengths(dataset, label):
    """
    Function that measures and prints token lengths for a dataframe column.
    """

    text_lengths = [len(word_tokenize(text)) for text in dataset[label]]

    print(f'\n{label} - minimum token length: ', min(text_lengths))
    print(f'{label} - mean token length: ', round(mean(text_lengths), 2))
    print(f'{label} - maximum token length: ', max(text_lengths))

    return text_lengths


def measure_sentence_counts(dataset, label):
    """
    Function that measures and prints sentence counts for a dataframe column.
    """

    sentence_counts = [len(sent_tokenize(text)) for text in dataset[label]]

    print(f'\n\n{label} - minimum sentence count: ', min(sentence_counts))
    print(f'{label} - mean sentence count: ', round(mean(sentence_counts), 2))
    print(f'{label} - maximum sentence count: ', round(max(sentence_counts)), '\n\n')

    return sentence_counts


def load_datasets(dataset_path):
    """
    Function that loads all the .csv datasets to a list of pandas dataframes.
    """

    greek_wiki = pandas.read_csv(os.path.join(dataset_path,'gr_wiki_postprocessed.csv'), index_col = False)
    greeksum = load_greeksum(dataset_path)
    
    return [greek_wiki, greeksum]


def load_greeksum(dataset_path):
    """
    Function which loads all greeksum splits for analysis.
    """

    train = pandas.read_csv(os.path.join(dataset_path, 'greeksum_train.csv'), index_col = False)
    test = pandas.read_csv(os.path.join(dataset_path, 'greeksum_test.csv'), index_col = False)
    val = pandas.read_csv(os.path.join(dataset_path, 'greeksum_valid.csv'), index_col = False)

    return pandas.concat([train, test, val], axis = 0, join = 'outer')


def produce_ngrams(word_list, n):
    """
    Function which produces n-grams.
    """
    return list(zip(*[word_list[i:] for i in range(n)]))


def produce_all_ngrams(text, n):
    """
    Function which generates all n-grams up to n.
    """

    word_list = word_tokenize(text)
    ngrams = [produce_ngrams(word_list, i) for i in range(1, n + 1)]

    return ngrams


def calc_mean_novel_ngrams_percentages(dataset, text_col, summary_col, n = 4):
    """
    Function that counts the mean percentage of novel summary n-grams that do not occur in the document of each dataset.
    """

    novel_ngrams_percentages = {
        f'{i}-grams': [] for i in range(1, n + 1)
    }
    
    for text, summary in zip(dataset[text_col], dataset[summary_col]):
        text_ngrams_lists = produce_all_ngrams(text, n)
        summary_ngrams_lists = produce_all_ngrams(summary, n)

        for i, (text_ngrams, summary_ngrams) in enumerate(zip(text_ngrams_lists, summary_ngrams_lists)):
            summary_ngrams_set = set(summary_ngrams)
            
            if len(summary_ngrams_set) == 0:
                novel_ngrams_percentage = 0
            else:
                novel_ngrams_percentage = len(summary_ngrams_set - set(text_ngrams)) / len(summary_ngrams_set)

            novel_ngrams_percentages[f'{i + 1}-grams'].append(novel_ngrams_percentage)

    mean_novel_ngrams_percentages = {
        f'{i}-grams': round(mean(novel_ngrams_percentages[f'{i}-grams']) * 100, 2) for i in range(1, n + 1)
    }

    return mean_novel_ngrams_percentages


def calc_vocabulary(dataset, text_col, summary_col):
    """
    Function that counts the vocabulary of each dataset.
    """

    text_vocab, summary_vocab = [], []
    
    for text, summary in zip(dataset[text_col], dataset[summary_col]):
        text_vocab.extend(word_tokenize(text))
        summary_vocab.extend(word_tokenize(summary))

    vocabulary_counts = {
        'documents': len(set(text_vocab)),
        'summaries': len(set(summary_vocab))
    }

    return vocabulary_counts


def load_local_dataset(local_dataset_dir: str) -> Tuple[HFDataset, HFDataset, HFDataset]:
    """
    Utility function which loads the required local dataset splits.
   
    Parameters
    ------------
    local_dataset_dir: the local dataset directory (str).

    Returns
    --------
    <object>: All dataset objects (Tuple[HFDataset, HFDataset, HFDataset])
    """
    train_dataset = load_dataset('csv', data_files = os.path.join(local_dataset_dir, 'gr_wiki_train.csv'), split = 'all')
    test_dataset = load_dataset('csv', data_files = os.path.join(local_dataset_dir, 'gr_wiki_test.csv'), split = 'all')
    validation_dataset = load_dataset('csv', data_files = os.path.join(local_dataset_dir, 'gr_wiki_val.csv'), split = 'all')

    return(train_dataset, test_dataset, validation_dataset)


def load_greeksum_dataset(local_dataset_dir: str) -> Tuple[HFDataset, HFDataset, HFDataset]:
    """
    Utility function which loads the required local dataset splits.
   
    Parameters
    ------------
    local_dataset_dir: the local dataset directory (str).

    Returns
    --------
    <object>: All dataset objects (Tuple[HFDataset, HFDataset, HFDataset])
    """
    train_dataset = load_dataset('csv', data_files = os.path.join(local_dataset_dir, 'greeksum_train.csv'), split = 'all')
    test_dataset = load_dataset('csv', data_files = os.path.join(local_dataset_dir, 'greeksum_test.csv'), split = 'all')
    validation_dataset = load_dataset('csv', data_files = os.path.join(local_dataset_dir, 'greeksum_valid.csv'), split = 'all')

    return(train_dataset, test_dataset, validation_dataset)


def load_abstractive_models(
        language_model_paths: List[str],
        device: str = 'cpu',
        local_files_only: bool = True
    ) -> Dict[str, Tuple[Model, Model]]:
    """
    Utility function which loads the selected abstractive models.
    
    Parameters
    ------------
    language_model_paths: List of huggingface model paths (List[str]).
    device: device to load and run models ['cpu', 'cuda:0'] (str).
    local_files_only: boolean flag which loads only a local model (bool).

    Returns
    --------
    <object>: All model objects (Dict[str, Tuple[Model, Model]]).
    """
    models = dict()

    for model_path in language_model_paths:
        
        # Load the tokenizer either from the HuggingFace model hub or locally.
        if local_files_only:
            tokenizer = AutoTokenizer.from_pretrained(f'{model_path}/tokenizer/', model_max_length = 1024, truncation = True, padding = 'max_length')
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length = 1024, truncation = True,  padding = 'max_length')

        # Load the pre-trained language model either from the HuggingFace model hub or locally.
        language_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only = local_files_only)
        
        # Send the model to the pre-specified device (cpu / gpu).
        language_model = language_model.to(device)

        # Assign the tokenizer and its language model to the corresponding model name key.
        models[model_path.split('\\')[1]] = (tokenizer, language_model)

    return models


def load_models(
        language_model_path: str,
        device: str = 'cpu',
        local_files_only: bool = False
    ) -> Tuple[Model, Model]:
    """
    Utility function which loads the required models.
    
    Parameters
    ------------
    language_model_path: path to huggingface model (str).
    device: device to load and run model ['cpu', 'cuda:0'] (str).
    local_files_only: boolean flag which loads only a local model (bool).

    Returns
    --------
    <object>: All model objects (Tuple[Model, Model]).
    """
    
    # Load the tokenizer either from the HuggingFace model hub or locally.
    if local_files_only:
        tokenizer = AutoTokenizer.from_pretrained(f'{language_model_path}/tokenizer/', model_max_length = 1024, truncation = True, padding = 'max_length')
    else:
        tokenizer = AutoTokenizer.from_pretrained(language_model_path, model_max_length = 1024, truncation = True,  padding = 'max_length')

    # Load the pre-trained language model either from the HuggingFace model hub or locally.
    language_model = AutoModelForSeq2SeqLM.from_pretrained(language_model_path, local_files_only = local_files_only)

    # Send the model to the pre-specified device (cpu / gpu).
    language_model = language_model.to(device)

    return (tokenizer, language_model)


def find_output_length(tokenizer: Model, train_dataset: HFDataset, validation_dataset: HFDataset, test_dataset: HFDataset) -> int:
    """
    The maximum total sequence length for output text after tokenization.
    Sequences longer than this will be truncated, sequences shorter will be padded.

    Parameters
    -----------
    tokenizer: huggingface tokenizer model (Model).
    train_dataset: HuggingFace training dataset (HFDataset).
    validation_dataset: HuggingFace validation dataset (HFDataset).
    test_dataset: HuggingFace testing dataset (HFDataset).
    
    Returns
    --------
    max_output_length: The maximum output length (int).
    """
    tokenized_outputs = concatenate_datasets([train_dataset, validation_dataset, test_dataset]).map(
        lambda x: tokenizer(x['abstract'], truncation = True), batched = True, remove_columns = ['article', 'abstract']
    )

    max_output_length = max([len(x) for x in tokenized_outputs['input_ids']])

    return max_output_length


def load_combined_dataset(local_dataset_dir: str) -> HFDataset:
    """
    Function which loads the training splits of GreekSUM and GreekWikipedia.
    Then it combines them into a single dataset, which is shuffled.

    Parameters
    -----------
    local_dataset_dir: the local dataset directory (str).
    
    Returns
    --------
    combined_dataset: The combined shuffled dataset (HFDataset).
    """
    # Load the training splits.
    greeksum_train = load_dataset('csv', data_files = os.path.join(local_dataset_dir, 'greeksum_train.csv'), split = 'all')
    greek_wiki_train = load_dataset('csv', data_files = os.path.join(local_dataset_dir, 'gr_wiki_train.csv'), split = 'all')

    # Remove unnecessary columns from Greek Wikipedia.
    greek_wiki_train = greek_wiki_train.remove_columns(['title', 'url'])

    # Shuffle the combined dataset.
    combined_dataset = concatenate_datasets([greeksum_train, greek_wiki_train]).shuffle(seed = 42)

    return combined_dataset


def save_tokenizer(tokenizer: Model, model_local_path: str):
    """
    Utility function which saves the model tokenizer locally.
   
    Parameters
    ------------
    tokenizer: huggingface tokenizer model (Model).
    model_local_path: path to local huggingface model (str).

    Returns
    --------
    <object>: All dataset objects (Tuple[HFDataset, HFDataset, HFDataset])
    """
    
    tokenizer.save_pretrained(model_local_path)
    return


def csv_to_txt(dataset: HFDataset, save_dir: str):
    """
    Utility function which saves each csv entry as a seperate csv file.

    Parameters
    -----------
    dataset: csv dataset (HFDataset).
    save_dir: Output file directory (str).
   
    Returns
    --------
    """
    
    for i, item in enumerate(dataset): 
        with open(os.path.join(save_dir, f'{i}.txt'), 'w', encoding = 'utf-8-sig', errors = 'ignore') as f:
            f.write(item['summary'])
    return
