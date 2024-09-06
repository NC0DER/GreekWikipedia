from src.crawl import *
from src.process import *
from src.analyze import *
from src.evaluation import *
from src.utils import load_models, load_local_dataset
from src.training import train_model


def main():

    crawl_greek_wikipedia()
    process_multiple_csv_files('data')
    postprocess('datasets')
    generate_dataset_split('datasets')
    
    analyze('datasets')
    analyze_dataset_length('datasets/gr_wiki_postprocessed.csv')

    train_dataset, _, val_dataset = load_local_dataset('datasets')
    tokenizer, model = load_models('google/umt5-small', device = 'cuda:0', local_files_only = False)
    train_model(train_dataset, val_dataset, tokenizer, model, output_dir = 'output')
    run_experiments()

    return


if __name__ == '__main__': main()
