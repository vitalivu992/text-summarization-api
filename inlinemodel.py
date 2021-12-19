import random
import time

from nlpconnector import tokenizer, device, dataset, model


def summary(text):
    # ARTICLE_TO_SUMMARIZE = dataset['test'][3]['article']
    # inputs = tokenizer([ARTICLE_TO_SUMMARIZE], truncation=True, return_tensors='pt')
    inputs = tokenizer(text, truncation=True, return_tensors='pt')
    inputs["input_ids"] = inputs["input_ids"].to(device)
    summary_ids = model.generate(inputs['input_ids'])
    # print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
    return " ".join(
        [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]) \
        .replace("<n>", "\n")


def example_data():
    i = random.randint(0, dataset['test'].num_rows)
    return (
        dataset['test'][i]['article'],
        dataset['test'][i]['highlights']
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
