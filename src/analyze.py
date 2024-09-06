import matplotlib.pyplot as plt

from tqdm import tqdm
from src.utils import *

def analyze(dataset_path):
    """
    Function which analyzes the datasets.
    """
    datasets = load_datasets(dataset_path)

    labels = ['Greek_Wikipedia', 'GreekSUM_abstract', 'GreekSUM_title']
    
    headers = {
        'Greek_Wikipedia':{'text': 'article', 'summary': 'summary'},
        'GreekSUM_abstract':{'text': 'article', 'summary': 'summary'},
        'GreekSUM_title':{'text': 'article', 'summary': 'title'}
    }
    
    print('Greek_Wikipedia-title: \n')
    wiki_title_lengths = measure_token_lengths(datasets[0], 'title')
    wiki_title_sentence_counts = measure_sentence_counts(datasets[0], 'title')
    
    wiki_title_ngrams_percentages = calc_mean_novel_ngrams_percentages(datasets[0], 'article', 'title')
    wiki_title_vocabulary_counts = calc_vocabulary(datasets[0], 'article', 'title')
    
    print(f'Greek_Wikipedia_title - n-grams percentages: {wiki_title_ngrams_percentages}')
    print(f'Greek_Wikipedia_title - vocabulary counts: {wiki_title_vocabulary_counts}')

    text_word_counts = []
    text_sentence_counts = []
    summary_word_counts = []
    summary_sentence_counts = []

    # Print the percentiles for the output lengths.
    for dataset_name, dataset in tqdm(zip(labels, datasets)):
        print('\n', dataset_name, '\n\n')
        text_word_count = measure_token_lengths(dataset, headers[dataset_name]['text'])
        summary_word_count = measure_token_lengths(dataset, headers[dataset_name]['summary'])
        text_sentence_count = measure_sentence_counts(dataset, headers[dataset_name]['text'])
        summary_sentence_count = measure_sentence_counts(dataset, headers[dataset_name]['summary'])

        text_word_counts.append(text_word_count)
        summary_word_counts.append(summary_word_count)
        text_sentence_counts.append(text_sentence_count)
        summary_sentence_counts.append(summary_sentence_count)

        ngrams_percentages = calc_mean_novel_ngrams_percentages(dataset, headers[dataset_name]['text'], headers[dataset_name]['summary'])
        vocabulary_counts = calc_vocabulary(dataset, headers[dataset_name]['text'], headers[dataset_name]['summary'])

        print(f'{dataset_name} - n-grams percentages: {ngrams_percentages}')
        print(f'{dataset_name} - vocabulary counts: {vocabulary_counts}')

    # Draw the boxplots and bar plot.
    plt.boxplot(text_word_counts[:2], showfliers = False)
    plt.xticks([1, 2], labels = ['Greek Wikipedia (Abstract)', 'GreekSUM (Abstract)'])
    plt.savefig(os.path.join('figures', 'text_word_counts_boxplots.svg'), format = 'svg', bbox_inches = 'tight')
    plt.clf()

    plt.boxplot(text_word_counts[:2], showfliers = False)
    plt.xticks([1, 2], labels = ['Greek Wikipedia', 'GreekSUM'])
    plt.savefig(os.path.join('figures', 'text_word_counts_boxplots_2.svg'), format = 'svg', bbox_inches = 'tight')
    plt.clf()

    plt.boxplot(summary_word_counts[:2], showfliers = False)
    plt.xticks([1, 2], labels = ['Greek Wikipedia (Abstract)', 'GreekSUM (Abstract)'])
    plt.savefig(os.path.join('figures', 'summary_word_counts_boxplots.svg'), format = 'svg', bbox_inches = 'tight')
    plt.clf()

    return
