import os.path
import random
import time

import datasets
import torch
# from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model_name = 'google/pegasus-cnn_dailymail'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

offline_model_dir = "./pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(offline_model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(offline_model_dir).to(device)

dataset_dir = "./cnn_dailymail"
if not os.path.exists(dataset_dir):
    dataset = datasets.load_dataset("cnn_dailymail", '3.0.0')
    print("Loading  dataset for the first time, saving to ", dataset_dir)
    dataset.save_to_disk(dataset_dir)
else:
    dataset = datasets.load_from_disk(dataset_dir)


def summary(text):
    # ARTICLE_TO_SUMMARIZE = dataset['test'][3]['article']
    # inputs = tokenizer([ARTICLE_TO_SUMMARIZE], truncation=True, return_tensors='pt')
    inputs = tokenizer(text, truncation=True, return_tensors='pt')
    inputs["input_ids"] = inputs["input_ids"].to(device)
    summary_ids = model.generate(inputs['input_ids'])
    # print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
    return " ".join(
        [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])\
        .replace("<n>", "\n")


def example_data():
    i = random.randint(0, dataset['train'].num_rows)
    return (
        dataset['train'][i]['article'],
        dataset['train'][i]['highlights']
    )


if __name__ == '__main__':
    example = example_data()
    print("### Example article")
    print(example[0])
    print("### Highlight")
    print(example[1])
    t0 = time.perf_counter()
    print("### Generated summary")
    print(summary(example[0]))
    print("### Time spent", time.perf_counter() - t0, " seconds")
