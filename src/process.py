import os
import pandas
import pathlib

from tqdm import tqdm
from sklearn.utils import shuffle
from src.utils import *


def process_multiple_csv_files(input_path):

    # Construct the input path.
    path = pathlib.Path(input_path)

    # Find all .csv files and their assosiated paths.
    csv_paths = [
        p.absolute()
        for p in path.iterdir()
        if p.is_file() and p.suffix == '.csv'
    ]

    # Append all dataframes to a list.
    df_list = []
    larger_summaries, empty_summaries, empty_articles = 0, 0, 0

    for csv_path in tqdm(csv_paths):
        # Each dataframe contains exactly one row.
        row = pandas.read_csv(csv_path, index_col = False).iloc[0]
        
        # If the summary does not exist, continue to the next row.
        if row['summary'] and not isinstance(row['summary'], str):
            empty_summaries += 1
            continue

        # If the article does not exist, continue to the next row.
        if row['article'] and not isinstance(row['article'], str):
            empty_articles += 1
            continue

        # Remove all section headers from the article.
        article = re.sub(r'==.*?==+', '', row['article'])

        # Remove extra whitespaces and unnecessary sections.
        summary = ' '.join(row['summary'].split())
        article = ' '.join(article.split())

        # The summary should always be smaller than the article.
        if summary.count(' ') >= article.count(' '):
            larger_summaries += 1
            continue
        
        # Create the new processed row as a dataframe of exactly one row.
        processed_row = pandas.DataFrame(
            zip([row['title']], [article], [summary], [row['url']]),
                columns = ['title','article', 'summary', 'url']
        )
        
        df_list.append(processed_row)

    print(f'Empty articles: {empty_articles}')
    print(f'Empty summaries: {empty_summaries}')
    print(f'Larger summaries: {larger_summaries}')

    # Make a new dataframe which combines the text from all processed dataframes.
    df = pandas.concat(df_list, axis = 0, join = 'outer')

    # Keep rows with text consisting of more than 25 words.
    df = df[df['article'].str.count(' ') > 25]

    # Remove duplicate rows.
    df = df.drop_duplicates(subset = 'url', keep = 'first')
    df = df.drop_duplicates(subset = 'title', keep = 'first')
    df = df.drop_duplicates(subset = 'article', keep = 'first')
    df = df.drop_duplicates(subset = 'summary', keep = 'first')

    print(f'Overall dataset length after preprocessing: {len(df)}')

    # Save the processed dataframe into a .csv dataset.
    df.to_csv('datasets/gr_wiki_all.csv', encoding = 'utf-8', index = False)

    return


def postprocess(dataset_path):

    dataset = pandas.read_csv(os.path.join(dataset_path, 'gr_wiki_all.csv'), index_col = False)
    titles, articles, summaries, urls = [], [], [], []
    
    for title, text, summary, url in tqdm(zip(dataset['title'], dataset['article'], dataset['summary'], dataset['url']), desc = 'Postprocessing Dataset...'):

        # Remove the summary from the first lines of the article.
        article = text.replace(summary, '')

        summary_tokens = word_tokenize(summary)
        new_article_tokens = word_tokenize(article)

        # Initialize counter variables.
        count_less_than_25, count_new_articles_less_than_summary = 0, 0 

        # If either the summary or the article have less than 25 tokens, continue.
        if not (len(summary_tokens) > 25 and len(new_article_tokens) > 25):
            count_less_than_25 += 1
            continue

        # If the article does not have more tokens than the summary, continue.
        if len(new_article_tokens) <= len(summary_tokens):
            count_new_articles_less_than_summary += 1
            continue
       
        titles.append(title)
        articles.append(article)
        summaries.append(summary)
        urls.append(url)

    # Create a new dataset.
    new_dataset = pandas.DataFrame({
        'title': titles,
        'article': articles,
        'summary': summaries,
        'url': urls
    })

    # Remove rows with empty articles or summaries.
    new_dataset = new_dataset[new_dataset['article'].notna()]
    new_dataset = new_dataset[new_dataset['summary'].notna()]
    
    # Save the postprocessed dataset into a csv file.
    new_dataset.to_csv(os.path.join('datasets', 'gr_wiki_postprocessed.csv'), encoding = 'utf-8', index = False)
    
    print('Postprocessed articles with less than 25 words: ', count_less_than_25)
    print('Postprocessed articles with text smaller than the summary: ', count_new_articles_less_than_summary)
    return


def generate_dataset_split(dataset_path):

    # Read the spreadsheet.csv
    df = pandas.read_csv(os.path.join(dataset_path, 'gr_wiki_postprocessed.csv'), index_col = False)
    
    # Shuffle the dataframe with a specific random seed.
    df = shuffle(df, random_state = 42)

    # Select 5000 (~5%) random samples for testing and validation respectively.
    # The rest of the samples are kept for training.
    test_df = df.iloc[:5000]
    val_df = df.iloc[5000:10000]
    train_df = df.iloc[10000:]

    # Save all dataset splits to .csv files.
    val_df.to_csv(os.path.join(dataset_path, 'gr_wiki_val.csv'), encoding = 'utf-8', index = False)
    test_df.to_csv(os.path.join(dataset_path, 'gr_wiki_test.csv'), encoding = 'utf-8', index = False)
    train_df.to_csv(os.path.join(dataset_path,'gr_wiki_train.csv'), encoding = 'utf-8', index = False)

    return
