import sys

import torch.cuda

"""
Tasks for this assignment
- For each dataset, create an n gram language model using NLTK's LM package as a baseline model
- Improve your n gram langauge models (Reduce perplexity) by using different types of smoothing, backoff, and interpolation
- For each language model, compute the perplexity of the test item. 
    - Whichever language model gives the lowest perplexity should be how you classify the test item
- For each language model, generate 5 samples of each author given the sample prompt you specify and compare them.
- Optional bonus points: implement n gram language models without using NLTK

MORE NOTES
- Authorlist is a list of file names to train the model on
- Test flag
    - True: use all data to train the model, then ouput classification results for each line in the given testfile (you may assume that each line of test file is an entire sentence)
    - False: extract development set (10%) from data, train model on 90% and run the model on develoment set and print results
- Each author has own nGram model
- 
"""
# Requirements
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, Pipeline
from datasets import load_dataset, Dataset
import evaluate
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline


class NGramModel:
    """Custom ngram model from nltk"""
    def __init__(self):
        self.model = None


class SequenceModel:
    """Auto model for sequence classification from hugging face"""
    def __init__(self, hf_model_name, authorlist, testflag=False):
        self.raw_data = self.create_dataset(authorlist)
        self.id2label = None
        self.label2id = None
        self.accuracy = evaluate.load('accuracy')

        self.model_name = hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_to_data(self):
        """ Function where the model gets trained"""
        # set to gpu if available
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        print(f'device: {device}')

    def preprocess_function(self,examples):
        return self.tokenizer(examples['text'], truncation=True)

    def compute_metrics(self, eval_pred):
        pass

    def split_data(self):
        pass
    
    def create_dataset(self, authorlist):
        seed = 19
        label2id = {}
        id2label = {}

        lines = []
        labels = []

        for i in range(len(authorlist)):
            author = authorlist[i].split("_")[0]
            label2id[author] = i
            id2label[i] = author
            with open(authorlist[i], "r", encoding="utf-8") as file:
                file_data = file.read()
                file_lines = file_data.splitlines()
            file.close()
            lines.extend(file_lines)
            labels.extend([i] * len(file_lines))

        # create dataset object from the files
        data = Dataset.from_dict({"text": lines, "label": labels})
        data = data.shuffle(seed)

        self.id2label = id2label
        self.label2id = label2id

        print("label2id: ", label2id)
        print("id2label: ", id2label)
        return data

    def evaluate(self):
        pass


def handler(args):
    pass


if __name__ == "__main__":
    print("Starting Program")
    args = sys.argv
    handler(args)

