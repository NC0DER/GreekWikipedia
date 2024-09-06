import os
import pathlib

from tqdm import tqdm
from statistics import mean
from transformers import pipeline
from evaluate import load
from typing import Tuple, TypeVar, List, Dict
from src.extractive import *
from src.utils import *

# Generic type class for model and dataset objects.
Model = TypeVar('Model')
HFDataset = TypeVar('HFDataset')


def abstractive_model_inference(text: str, tokenizer: Model, model: Model) -> str:
    """
    This function summarizes a text using the selected Huggingface model.

    Parameters
    -----------
    text: text to be summarized (str).
    tokenizer: the tokenizer model (Model).
    model: the Huggingface model (Model).
   
    Returns
    summary: An abstractive summary (str).
    --------
    """
    # Initialize the summarization pipeline.  
    summarizer = pipeline('summarization', model = model, tokenizer = tokenizer, max_new_tokens = 128, truncation = True, repetition_penalty = 1.2) # device = 'cuda:0',
    summary = summarizer('summarize: ' + text)

    return summary[0]['summary_text']


def produce_summaries(dataset: HFDataset, save_dir: str, model_names: List[str], 
                      abstractive_models: Dict[str, Tuple[Model, Model]]):
    """
    This function produces summaries for each dataset item 
    and saves these summaries as separate .txt files in a separate directory for each model.

    Parameters
    -----------
    dataset: Dataset to produce summaries from (HFDataset).
    save_dir: Save directory for all model produced summaries (str).
    model_names: List of model names (List[str]).
    abstractive_models: Dictionary of summarization models (Dict[str, Tuple[Model, Model]]).

    Returns
    --------
    None.
    """

    # Iterate each dataset item (article).
    for i, item in tqdm(enumerate(dataset), desc = 'Producing summaries'):

        # Produce a summary for each model.
        for model_name in model_names:

            if model_name == 'textrank': 
                summary = textrank_gr(item['article'], 4)
            elif model_name == 'lexrank': 
                summary = lexrank_gr(item['article'], 4)
            elif model_name == 'lead': 
                summary = lead_gr(item['article'], 4)
            elif model_name == 'greek-umT5-small-hybrid-epoch-8' or model_name == 'greek-umT5-small-hybrid-epoch-10':
                summary = abstractive_model_inference(
                    reduce_text(item['article'], abstractive_models[model_name][0]), 
                    *abstractive_models[model_name]
                )
            else:
                summary = abstractive_model_inference(
                    item['article'], *abstractive_models[model_name]
                )

            # Save the summary in a separate file for each dataset entry.
            output_path = os.path.join(save_dir, model_name)
            pathlib.Path(output_path).mkdir(parents = True, exist_ok = True)
            
            with open(os.path.join(output_path, f'{i}.txt'),
                    'w', encoding = 'utf-8-sig', errors = 'ignore') as f:
                f.write(summary)
    return


def evaluate(produced_path: str, reference_path: str,
             dataset_length: int, slice_size: int) -> Dict[str, int]:
    """
    This function compares the produced summaries of each method
    against the reference summaries. The evaluation metrics are ROUGE and BERTScore. 
    For each score we calculate the macro(mean) F1 score.
 
    Parameters
    -----------
    produced_path: Path of the machine produced summaries (str).
    reference_path: Path of the human written summaries (str).
    dataset_length: The length of the dataset (int).
    slice_size: The size of each dataset slice passed to the BERTScore scorer, 
    needs to be a whole multiple of dataset_length (int).
   
    Returns
    --------
    metrics: A dictionary which contains the values of each evaluation metric (Dict[str, int]).
  
    """
    # Initialize the metric scores.
    metric_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bertscore': 0.0}
    
    # Initialize the evaluation metrics.

    bertscore = load('bertscore') 
    rouge = load('rouge')

    # Initialize the prediction and reference lists.
    predictions, references = [], []

    # Read the summary text files and pass them into the lists.
    for i in tqdm(range(dataset_length), desc = 'Loading files for evaluation...'):
        with open(os.path.join(produced_path, f'{i}.txt'), 'r', encoding = 'utf-8-sig', errors = 'ignore') as pred, \
            open(os.path.join(reference_path, f'{i}.txt'), 'r', encoding = 'utf-8-sig', errors = 'ignore') as ref:
            predictions.append(pred.read())
            references.append(ref.read())

    # Split the dataset into slices and calculate the BERTScore for each one.
    for i in tqdm(range(dataset_length // slice_size), desc = f'Computing BERTScore metric scores...'):
        metric_scores['bertscore'] += mean(bertscore.compute(
            predictions = predictions[i * slice_size:i * slice_size + slice_size],
            references = references[i * slice_size:i * slice_size + slice_size],
            lang = 'el', device = 'cpu')['f1']
        )

    if (mod := dataset_length % slice_size):
        print('Computing BERTScore metric scores for the remainder documents...')
        metric_scores['bertscore'] += mean(bertscore.compute(
            predictions = predictions[(i + 1) * slice_size:(i + 1) * slice_size + mod],
            references = references[(i + 1) * slice_size:(i + 1) * slice_size + mod],
            lang = 'el', device = 'cpu')['f1']
        )
        i += 1
    
    # Calculate the macro BERTScore.
    metric_scores['bertscore'] = metric_scores['bertscore'] / (i + 1)

    # Compute the ROUGE metric score for each ROUGE metric.
    for i in tqdm(range(dataset_length), desc = 'Computing ROUGE metric scores...'):   
        
        # For each non-empty pair of summaries compute the ROUGE metric scores.
        if predictions[i] and not predictions[i].isspace():
            scores = rouge.compute(predictions = [predictions[i]], references = [references[i]], tokenizer = lambda x: x.split())
            metric_scores['rouge1'] += scores['rouge1']
            metric_scores['rouge2'] += scores['rouge2']
            metric_scores['rougeL'] += scores['rougeL']
            
 
    # Calculate the macro ROUGE metric scores.
    # Empty documents are taken into account by dividing with the entire dataset length. 
    metric_scores['rouge1'] = metric_scores['rouge1'] / dataset_length
    metric_scores['rouge2'] = metric_scores['rouge2'] / dataset_length
    metric_scores['rougeL'] = metric_scores['rougeL'] / dataset_length

    print(metric_scores)
    return metric_scores


def run_experiments():

    # Initialize a list with the selected abstractive models.
    model_paths = [
        'models\greek-umT5-base-epoch-10'
    ]

    # Set the selected summarization methods.
    model_names = [*map(lambda x: x.split('\\')[1], model_paths)]

    models = load_abstractive_models(
        language_model_paths = model_names, 
        device = 'cuda:0',
        local_files_only = True 
    )
    
    # Load the testing dataet (.csv)
    _, test_dataset, _ = load_local_dataset('datasets')
    
    # Set the output path for the produced summaries.
    output_path = 'output'

    # Produce summaries for each method.
    produce_summaries(test_dataset, output_path, model_names, models)
    
    # Create a directory for the reference summaries and save each one in a separate .txt file. 
    reference_path = 'output\greek_wiki_test'
    pathlib.Path(reference_path).mkdir(parents = True, exist_ok = True)
    csv_to_txt(test_dataset, reference_path)

    # Evaluate the selected methods against the reference summaries.
    for method in model_names:
        print(f'\n{method}:')
        print(evaluate(os.path.join(output_path, method), reference_path,  dataset_length = 5000, slice_size = 1000))

    return
