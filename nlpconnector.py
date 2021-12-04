import os

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
