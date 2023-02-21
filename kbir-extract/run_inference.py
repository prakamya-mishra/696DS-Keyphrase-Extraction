from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoConfig
)
import numpy as np
from datasets import load_dataset
import evaluate

from torch.utils.data import DataLoader
import torch

from tqdm import tqdm

import logging
import os, sys

def main():
    log_filename = 'log.txt'
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join('.', log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    #model_name = "bloomberg/KBIR"
    model_name = "roberta-base"
    max_len = 512
    batch_size = 8
    device = torch.device("cuda")

    logge.info('Running model: {}'.format(model_name))

    # Dataset parameters
    dataset_full_name = "midas/kpcrowd"
    dataset_subset = "extraction"
    dataset_document_column = "document"
    dataset_biotags_column = "doc_bio_tags"

    logger.info('Dataset: {}'.format(dataset_full_name))

    # Labels
    label_list = ["B", "I", "O"]
    lbl2idx = {"B": 0, "I": 1, "O": 2}
    idx2label = {0: "B", 1: "I", 2: "O"}

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    def preprocess_fuction(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(
            all_samples_per_split[dataset_document_column],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=max_len,
        )
        total_adjusted_labels = []
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split[dataset_biotags_column][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if wid is None:
                    adjusted_label_ids.append(lbl2idx["O"])
                elif wid != prev_wid:
                    i = i + 1
                    adjusted_label_ids.append(lbl2idx[existing_label_ids[i]])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(
                        lbl2idx[
                            f"{'I' if existing_label_ids[i] == 'B' else existing_label_ids[i]}"
                        ]
                    )

            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples["labels"] = total_adjusted_labels
        return_keys = ['input_ids', 'attention_mask', 'labels']
        return {k:tokenized_samples[k] for k in return_keys}

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        #if 'cpu' in device:
        #    y_pred = predictions.detach().clone().numpy()
        #    y_true = references.detach().clone().numpy()
        #else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def tokenize_and_align_labels(all_samples_per_split):
        tokenized_inputs = tokenizer.batch_encode_plus(
            all_samples_per_split[dataset_document_column],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=max_len,
        )
        labels = []
        for i, label in enumerate(all_samples_per_split[dataset_biotags_column]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  
                if word_idx is None:
                    label_ids.append(-100) # Set the special tokens to -100.
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(lbl2idx[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Load dataset
    dataset = load_dataset(dataset_full_name, dataset_subset)

    # Preprocess dataset
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names['train'])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    logger.info(tokenized_dataset)

    # Dataloader
    #dataloader_train = DataLoader(tokenized_dataset['train'], batch_size=batch_size)
    dataloader_test = DataLoader(tokenized_dataset['test'], batch_size=batch_size)

    #load model
    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model = torch.nn.DataParallel(model) # optional

    # Metrics
    metric = evaluate.load("seqeval")

    model.eval()
    dataloader_test = tqdm(dataloader_test)
    for step, batch in enumerate(dataloader_test):    
        params = {'input_ids':batch['input_ids'].to(device), 'attention_mask':batch['attention_mask'].to(device), 'labels':batch['labels'].to(device)}
        with torch.no_grad():
            outputs = model(**params)
        pred = outputs.logits.argmax(dim=-1)
        pred_bio, labels_bio = get_labels(pred, batch['labels'])
        metric.add_batch(predictions=pred_bio,references=labels_bio,)

    logger.info(metric.compute())
        
if __name__ == "__main__":
    main()