from transformers import pipeline, AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset
from math import ceil
from glob import glob
import re
import json
from collections import defaultdict
import os
import subprocess
from tqdm.auto import tqdm

def batches(dataset, batch_size):
        for batch in tqdm(range(0, len(dataset), batch_size), total=ceil(len(dataset) / batch_size)):
            yield dataset[batch: min(batch+batch_size, len(dataset))]

# dataset = load_dataset('squad_v2', split='validation')

def evaluate_answerability_classification(model, dataset, batch_size, max_seq_len=512, sep=None):
    
    sep = pipe.tokenizer.sep_token if not sep else sep
    
    def prepare_example(example):
        # only for electra for roberta shoueld have different SEP
        example['input'] = example['question'] + sep + example['context']
        example['label'] = int(bool(len(example['answers']['text'])))
        return example
    
    def filter_exmpale(example):
        return len(tokenizer(example['input'])['input_ids']) <= 512
    
    dataset = dataset.map(prepare_example).filter(filter_exmpale)
    
    answerable_total = 0
    answerable_correct = 0
    unanswerable_total = 0
    unanswerable_correct = 0
    
    for batch in batches(dataset, batch_size):
        preds = model(batch['input'])
        
        for pred, label in zip(preds, batch['label']):
            pred_label = model.model.config.label2id[pred['label']]
            if label:
                answerable_total += 1
                if pred_label:
                    answerable_correct += 1
            else:
                unanswerable_total += 1
                if not pred_label:
                    unanswerable_correct += 1      

    res = {
        'answerable_score': (answerable_correct * 1.0) / answerable_total,
        'unanswerable_score': (unanswerable_correct * 1.0) / unanswerable_total,
        'total_score':  ((answerable_correct + unanswerable_correct) * 1.0) / (answerable_total + unanswerable_total),
        'answerable_total':answerable_total,
        'unanswerable_total':unanswerable_total,
        'total':answerable_total + unanswerable_total
    }
    return res

# dataset = load_dataset('squad_v2', split='validation')

def evaluate_answerability_squad(model, dataset, batch_size, max_seq_len=512):
        
    answerable_total = 0
    answerable_correct = 0
    unanswerable_total = 0
    unanswerable_correct = 0
    
    for batch in batches(dataset, batch_size):
        preds = model(question=batch['question'],  context=batch['context'], handle_impossible_answer=True, topk=1, max_seq_len=max_seq_len)
        
        for pred, answers in zip(preds, batch['answers']):
            if answers['text']:
                answerable_total += 1
                if pred['answer']:
                    answerable_correct += 1
            else:
                unanswerable_total += 1
                if not pred['answer']:
                    unanswerable_correct += 1      

    res = {
        'answerable_score': (answerable_correct * 1.0) / answerable_total,
        'unanswerable_score': (unanswerable_correct * 1.0) / unanswerable_total,
        'total_score':  ((answerable_correct + unanswerable_correct) * 1.0) / (answerable_total + unanswerable_total),
        'answerable_total':answerable_total,
        'unanswerable_total':unanswerable_total,
        'total':answerable_total + unanswerable_total
    }
    return res



# hans = load_dataset('hans', split='validation')
# dataset = hans.filter(lambda e: e['heuristic'] == 'lexical_overlap')
#

def evaluate_entailment_classification(model, dataset, tokenizer=None, batch_size=1000, sep=None, entail_index=0):
        
    try:
        pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer if tokenizer != model else model)
    except:
        pipe = pipeline("sentiment-analysis", model=model)
        
    sep = pipe.tokenizer.sep_token if not sep else sep
    target_label = pipe.model.config.id2label[entail_index]

    def make_inputs(example):
        example['input'] = example['premise'] + sep + example['hypothesis']
        return example

    dataset = dataset.map(make_inputs)
    
    entailed_sum = 0
    entailed_count = 0
    non_sum = 0
    non_count = 0
    huristic = 0
    
    for batch in batches(dataset, batch_size):
        preds = pipe(batch['input'])
        for pred, label in zip(preds, batch['label']):
            pred = 0 if pred['label'] == target_label else 1
            if pred == 0:
                huristic += 1
            if label == 0:
                entailed_sum += int(label == pred)
                entailed_count += 1
            else:
                non_sum += int(label == pred)
                non_count += 1
    res = {
        'entailed score': entailed_sum / entailed_count,
        'non entailed score': non_sum / non_count,
        'total score':  (entailed_sum + non_sum) / (non_count + entailed_count),
        'huristic utilization score': huristic / (non_count + entailed_count),
    }
    return res


# dataset = load_dataset("csv", data_files="paws_qqp_test.tsv", delimiter='\t')['train']

def evaluate_paraphrase_detection(model, dataset, tokenizer=None, batch_size=100, sep=None, entail_index=0, regression=False):
    
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer if tokenizer != model else model)
    sep = pipe.tokenizer.sep_token if not sep else sep
    target_label = pipe.model.config.id2label[entail_index]
    

    def make_inputs(example):
        example['input'] = example['sentence2'] + sep + example['sentence1']
        return example

    dataset = dataset.map(make_inputs)
    
    def batches(dataset, n):
        for batch in tqdm(range(0, len(dataset), batch_size), total=int(len(dataset) / n)):
            yield dataset[batch: min(batch+batch_size, len(dataset))]

    
    not_duplicate_sum = 0
    not_duplicate_count = 0
    duplicate_sum = 0
    duplicate_count = 0
    
    for batch in batches(dataset, batch_size):
        preds = pipe(batch['input'])
        for pred, label in zip(preds, batch['label']):
            if regression:
                 pred = 0 if pred['score'] > 0.5 else 1
            else:
                pred = pipe.model.config.label2id[pred['label']]
            if label == 0:
                not_duplicate_sum += int(int(label) == pred)
                not_duplicate_count += 1
            else:
                duplicate_sum += int(label == pred)
                duplicate_count += 1
    res = {
        'not duplicate score': not_duplicate_sum / not_duplicate_count,
        'duplicate score': duplicate_sum / duplicate_count,
        'total score':  (not_duplicate_sum + duplicate_sum) / (not_duplicate_count + duplicate_count),
    }
    return res
