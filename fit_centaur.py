from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import gc
from trl import DataCollatorForCompletionOnlyLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    def split_array_by_bool_vector(array, bool_vector):
        if len(array) != len(bool_vector):
            raise ValueError("Array and boolean vector must have the same length.")
        split_indices = np.where(bool_vector)[0]
        return np.split(array, split_indices)[:-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = 32768,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)
    print(l_id, flush=True)
    print(r_id, flush=True)

    train_data = load_dataset("marcelbinz/Psych-101")
    train_data = train_data['train'].filter(lambda example: example['experiment'] == args.experiment)
    test_data = load_dataset("marcelbinz/Psych-101-test")
    test_data = test_data['test'].filter(lambda example: example['experiment'] == args.experiment)
    dataset = concatenate_datasets([train_data, test_data])

    def tokenization(example):
        print(example)
        tokenized = tokenizer(example['text'])
        tokenized['participant'] = int(example['participant'])
        return tokenized

    dataset = dataset.map(tokenization).sort('participant')
    print(dataset)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "participant"])

    dataloader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=collator, batch_size=1)

    nlls = []
    with torch.no_grad():
        for data_part in dataloader:
            print(data_part['participant'])

            model_outputs = model(data_part.input_ids.to(device), data_part.attention_mask.to(device), return_dict=True)
            targets_ids = data_part.labels[0, 1:].detach().cpu()

            nll = torch.nn.functional.cross_entropy(model_outputs.logits[0, :-1].detach().cpu(), targets_ids, reduction='none')
            nll = nll[targets_ids != -100]
            nlls.append(nll)

            del targets_ids
            del model_outputs
            torch.cuda.empty_cache()
            gc.collect()

    torch.save(nlls, 'data/log_likelihoods_' + args.model.replace('/', '-') +  '.pth')
