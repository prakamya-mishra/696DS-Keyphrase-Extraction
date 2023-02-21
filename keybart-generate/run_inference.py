from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig
)
from datasets import load_dataset
import evaluate

from torch.utils.data import DataLoader
import torch
import nltk
import numpy as np

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
    
    model_name = "bloomberg/KeyBART"
    max_len = 512
    batch_size = 8
    device = torch.device("cuda")

    logger.info("Model: {}".format(model_name))

    # Dataset parameters
    dataset_full_name = "midas/kpcrowd"
    dataset_subset = "generation"
    dataset_document_column = "document"
    dataset_summary_column = "extractive_keyphrases"

    logger.info("Dataset: {}".format(dataset_full_name))


    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    
    
    def preprocess_function(examples):
        # remove pairs where at least one record is None
        cnt = 0
        inputs, targets = [], []
        for i in range(len(examples[dataset_document_column])):
            if examples[dataset_document_column][i] and examples[dataset_summary_column][i]:
                inputs.append(examples[dataset_document_column][i])
                if examples[dataset_summary_column][i] == 0:
                    cnt += 1
                targets.append(examples[dataset_summary_column][i])
        model_inputs = tokenizer.batch_encode_plus(inputs, max_length=max_len, padding='max_length', truncation=True, is_split_into_words=True,)

        
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_len, padding='max_length', truncation=True, is_split_into_words=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    # Load dataset
    dataset = load_dataset(dataset_full_name, dataset_subset)
    
    # Preprocess dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names['train'])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print(tokenized_dataset)

    # Dataloader
    #dataloader_train = DataLoader(tokenized_dataset['train'], batch_size=batch_size)
    dataloader_test = DataLoader(tokenized_dataset['test'], batch_size=batch_size)

    #load model
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    print('====model vocab size======')
    print(model.config.vocab_size)
    print('=========================')

    print('====len of tokenizer======')
    print(len(tokenizer))
    print('=========================')

    
    model.to(device)
    #model = torch.nn.DataParallel(model)

    # Metrics
    metric = evaluate.load("rouge")

    model.eval()
    dataloader_test = tqdm(dataloader_test)
    for step, batch in enumerate(dataloader_test):    
        gen_kwargs = {"max_length":15, "num_beams":3}
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                **gen_kwargs,
            )
        
            generated_tokens = generated_tokens.cpu().numpy()
            labels = batch['labels'].numpy()


            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )

    result = metric.compute(use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}

    logger.info(result)

    

if __name__ == "__main__":
    main()