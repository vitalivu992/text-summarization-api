import random

import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

model_name = 'google/pegasus-cnn_dailymail'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", '3.0.0')

ARTICLE_TO_SUMMARIZE = dataset['test'][3]['article']
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], truncation=True, return_tensors='pt')
inputs["input_ids"] = inputs["input_ids"].to(device)
summary_ids = model.generate(inputs['input_ids'])

print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


def example():
    i = random.nextint(dataset['train'].num_rows)
    return (
        dataset['train'][i]['article'],
        dataset['train'][i]['highlights']
    )


if __name__ == '__main__':
    print(example())
