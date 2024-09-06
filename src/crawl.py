import time
import wikipedia
import pandas
import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def crawl_greek_wikipedia():

    # Select the greek wikipedia.
    wikipedia.set_lang('el')

    # Read all greek wikipedia titles from file, disambiguation articles are manually excluded.
    with open('elwiki-latest-all-titles(processed)', 'r', encoding = 'utf-8-sig') as f:
        titles = f.read()
    
    # Split the titles text into a list of titles.
    titles = titles.split('\n')

    # Set the indices for start to end.
    start, end = 0, len(titles)

    # Retrieve the wikipedia page based on the current title, 
    # unless it is a disambiguation page or the page could not be accessed.
    pause_time = 4.0
    for i in tqdm(range(start, end)):

        s = default_timer()
        try:
            page = wikipedia.page(titles[i])
        except (wikipedia.DisambiguationError, wikipedia.exceptions.PageError) as err:
            e = default_timer()
            download_time = e - s
            if download_time < pause_time:
                time.sleep(pause_time - download_time)
            continue
        except wikipedia.exceptions.WikipediaException as error:
            error_msg = repr(error)
            if 'Regular expression is too complex' in error_msg:
                e = default_timer()
                processing_time = e - s
                if processing_time < pause_time:
                    time.sleep(pause_time - processing_time)
                continue
            else:
                print(error_msg)
                return

        e = default_timer()
        download_time = e - s

        # Create a pandas dataframe from each wikipedia page and save it to a csv.
        s = default_timer()
        wiki_df = pandas.DataFrame(zip([page.title], [page.content], [page.summary], [page.url]),
        columns = ['title','article', 'summary', 'url'])
        wiki_df.to_csv(rf'data\wiki_{i}.csv', encoding = 'utf-8', index = False)
        e = default_timer()
        storing_time = e - s

        # Rate limit the crawling to every pause_time amount of seconds.
        if download_time + storing_time < pause_time:
            time.sleep(pause_time - download_time - storing_time)


def analyze_dataset_length(local_dataset_dir):
    
    # Load the dataset using the transformers dataset library.
    dataset = load_dataset('csv', data_files = local_dataset_dir, split = 'all')

    # Load the T5 tokenizer.
    tokenizer = AutoTokenizer.from_pretrained('google/umt5-small', model_max_length = 1024, truncation = True, padding = 'max_length')

    # Tokenize the inputs and outputs.
    tokenized_inputs = dataset.map(
        lambda x: tokenizer(x['article'], truncation = True), batched = True, remove_columns = ['article']
    )
    tokenized_outputs = dataset.map(
        lambda x: tokenizer(x['summary'], truncation = True), batched = True, remove_columns = ['summary']
    )

    # Calculate the input and outputs lengths of each dataset row.
    input_lengths = [len(x) for x in tokenized_inputs['input_ids']]
    output_lengths = [len(x) for x in tokenized_outputs['input_ids']]

    # Print the percentiles for the output lengths.
    output_lengths = numpy.array(output_lengths)
    percentiles = [
        f'{p}%: {numpy.percentile(output_lengths, p)}' 
        for p in range(0, 125, 25)
    ]
    print(percentiles)

    # Draw the boxplots and histogram.
    plt.boxplot([input_lengths, output_lengths])
    plt.savefig('boxplots.png')
    plt.clf()
    plt.hist(output_lengths)
    plt.savefig('output_tokens_hist.png')

    return
