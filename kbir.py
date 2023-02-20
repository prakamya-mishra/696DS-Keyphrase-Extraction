from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import evaluate
from tqdm.auto import tqdm
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The name of the huggingface model to use.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_subset_name",
        type=str,
        default="raw",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The name of the text column in the dataset.",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The name of the label column in the dataset.",
    )
    parser.add_argument(
        "--padding_type",
        type=str,
        default="max_length",
        help="Type of padding.",
    )
    parser.add_argument(
        "--padding_length",
        type=int,
        default=512,
        help="Length of padding.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size to use during evaluation.",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default=None,
        help="Name of the result file.",
    )
    args=parser.parse_args()
    return args


def main():
    args = parse_args()

    text_column_name = args.text_column_name
    label_column_name = args.label_column_name
    dataset_full_name = args.dataset_name
    dataset_subset = args.dataset_subset_name
    model_name = args.model_name
    
    dataset = load_dataset(dataset_full_name, dataset_subset)
    def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

    label_list = get_label_list(dataset["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
            if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
              label_list = [model.config.id2label[i] for i in range(num_labels)]
              label_to_id = {l: i for i, l in enumerate(label_list)}

    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    padding = args.padding_type
    max_length = args.padding_length
    def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                max_length=max_length,
                padding=padding,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )

            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                      label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                      label_ids.append(label_to_id[label[word_idx]])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                      label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
    processed_dataset = dataset.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=dataset["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    eval_dataset = processed_dataset["test"]
    # eval_dataset_sample40 = Dataset.from_dict(eval_dataset[:40])
    # eval_dataset_sample40

    eval_batch_size=args.eval_batch_size
    data_collator = default_data_collator
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_batch_size)

    metric = evaluate.load("seqeval")
    accelerator = Accelerator()
    device = accelerator.device
    model, eval_dataloader= accelerator.prepare(model, eval_dataloader)

    def get_labels(predictions, references):
            # Transform predictions and references tensos to numpy arrays

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

    def compute_metrics():
            results = metric.compute()
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    model.eval()
    samples_seen = 0

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        # if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
        #     predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        #     labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions_gathered = predictions_gathered[: len(eval_dataloader.dataset) - samples_seen]
                labels_gathered = labels_gathered[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += labels_gathered.shape[0]
        preds, refs = get_labels(predictions_gathered, labels_gathered)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    eval_metric = compute_metrics()
    accelerator.print(eval_metric)
    filename=args.model_name+'_'+args.dataset_name+'.json'
    evaluate.save('./Results/'+filename, **eval_metric)

if __name__ == "__main__":
    main()