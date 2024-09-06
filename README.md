[![Python-Versions](https://img.shields.io/badge/python-3.10-blue.svg)]()
[![Open in HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Open_in_HuggingFace-orange)](https://huggingface.co/IMISLab/)
[![Software-License](https://img.shields.io/badge/License-Apache--2.0-green)](https://github.com/NC0DER/LMRank/blob/main/LICENSE)

# GreekWikipedia
This repository hosts code for the paper:
* [Giarelis, N., Mastrokostas, C., & Karacapilidis, N. (2024). Greek Wikipedia: A Study on Abstractive Summarization]()


## About
This repository stores the data crawling and processing code for the `GreekWikipedia` dataset, as well as the training and evaluation code for the proposed models.
The proposed models were trained and evaluated on `GreekWikipedia`.
The dataset and best-performing models are hosted on [HuggingFace](https://huggingface.co/IMISLab).


## Installation
```
pip install requirements.txt
```

## Example Code
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = 'IMISLab/GreekWiki-umt5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name) 

summarizer = pipeline(
    'summarization',
    device = 'cpu',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 220,
    truncation = True
)
    
text = 'Η Ανδρίτσα είναι οικισμός στο νοτιοδυτικό τμήμα του νομού Αργολίδας, δίπλα στα όρια με τον νομό Αρκαδίας. Βρίσκεται στις νοτιοανατολικές υπώρειες του Παρθενίου όρους και στις όχθες του μικρού ποταμού Ξαβριού, σε μέσο σταθμικό υψόμετρο 300. Απέχει 28 χλμ. περίπου ΝΔ. του Ναυπλίου. Η τοπική κοινότητα Ανδρίτσας είναι χαρακτηρισμένη ως αγροτικός ημιορεινός οικισμός, με έκταση 19,304 χμ² (2011). Ο πληθυσμός της Ανδρίτσας διπλασιάστηκε μεταξύ του 1879 και του 1889 αλλά έπειτα σταθεροποιήθηκε μέχρι και το 1961. Έκτοτε έχει συρρικνωθεί σημαντικά. Ο οικισμός αναγνωρίστηκε το 1879 και προσαρτήθηκε στον δήμο Υσιών του νομού Αργολίδος & Κορινθίας. Το 1899 εντάχθηκε στον νομό Αργολίδας και, το 1909, πάλι στον νομό Αργολίδος & Κορινθίας. Το 1932 ορίστηκε έδρα της κοινότητας Ανδρίτσας και το 1949 υπήχθη οριστικά στον νομό Αργολίδας. Με το ΦΕΚ 244Α - 04/12/1997 αποσπάστηκε από την κοινότητα Ανδρίτσας και προσαρτήθηκε στον δήμο Λέρνας. Με το ΦΕΚ 87Α - 07/06/2010 αποσπάστηκε από τον δήμο Λέρνας και προσαρτήθηκε στον δήμο Άργους-Μυκηνών. Η Ανδρίτσα είχε παλαιότερα σιδηροδρομικό σταθμό στη γραμμή Κορίνθου-Καλαμάτας. Στις υπώρειες του όρους Ζάβιτσα βρίσκεται το «Σπήλαιο Ανδρίτσας», η εξερεύνηση του οποίου ξεκίνησε στις αρχές του 2004 από την Εφορεία Παλαιοανθρωπολογίας-Σπηλαιολογίας Νότιας Ελλάδας.'
output = summarizer('summarize: ' + text)
print(output[0]['summary_text'])
```

## Citation
The model has been officially released with the article:  
[Giarelis, N., Mastrokostas, C., & Karacapilidis, N. (2024). Greek Wikipedia: A Study on Abstractive Summarization]().  
If you use the code, dataset or models, please cite the following:

```
TBA
```

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Charalampos Mastrokostas (cmastrokostas@ac.upatras.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
