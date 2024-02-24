# -*- coding: utf-8 -*-
import numpy as np
import random
import sys
from nltk.lm import MLE, KneserNeyInterpolated, Vocabulary
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.util import bigrams, ngrams
from nltk.tokenize import word_tokenize

# Disciminative
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, Pipeline
from datasets import load_dataset, Dataset
import evaluate
import torch
import ssl
import time

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk as nltk



class GenerativeModel:
    def __init__(self, al, flag, testfile=None, num_grams=2, print_misclassification=False) -> None:
        self.author_list = al
        self.test_flag = flag
        self.data = None
        self.vocab = None
        self.models = None
        self.testfile = testfile
        self.num_grams = num_grams
        self.print_misclassification = print_misclassification

        # Populate the models
        self.init_data()
        self.train_models()


    def init_data(self):
        data = {}
        nltk.download('punkt')

        unknown_token = '<unk>'
        vocab = []
        vocab.append(unknown_token)

        for filename in self.author_list:
            tokens = []
            author = filename.split("_")[0]
            with open(filename, 'r', encoding='utf-8') as fp:
                for line in fp:
                    if len(line) > 1:
                        line = word_tokenize(line.strip())
                        tokens.append(line)
                        vocab.append(line)
            data[author] = tokens

        if not self.test_flag:
            print("splitting into training and development...")

        for author, tokens in data.items():
            if not self.test_flag:
                index = int(len(tokens) * .9)
                split = {"train" : tokens[:index], "test" : tokens[index:]}
            else:
                split = {"train" : tokens, "test" : []}
            data[author] = split

        _, vocab = padded_everygram_pipeline(self.num_grams, vocab)

        vocab = list(vocab)

        for author, split in data.items():
            train_temp, _ = padded_everygram_pipeline(self.num_grams, split['train'])
            data[author]['train'] = list(train_temp)
            #print(f"{author} train length: {len(data[author]['train'])}")
            #print(f"{author} test length: {len(data[author]['test'])}")

        self.data = data
        self.vocab = Vocabulary(vocab, unk_cutoff=1)

    def train_models(self):
        start = time.time()
        models = {}
        print("training LMs... (this may take a while)")
        for author, split in self.data.items():
            model = KneserNeyInterpolated(self.num_grams, discount=0.9)
            model.fit(split['train'], self.vocab)
            models[author] = model
        print(f"Training time: {(time.time()-start):.2f} seconds")
        self.models = models

    def apply_models(self,sentence) -> str:
        curr_author = None
        perplex = 999999
        # test = list(ngrams(pad_both_ends(sentence, 2), self.num_grams))
        test = list(ngrams(pad_both_ends(sentence, 2), self.num_grams))
        if test:
            for author, model in self.models.items():
                score = model.perplexity(test)
                if score < perplex:
                    perplex = score
                    curr_author = author
        return curr_author

    def evaluate(self):
        if not self.test_flag:
            print("Results on dev set:")
            for author, split in self.data.items():
                correct = 0
                for sentence in split['test']:
                    predicted = self.apply_models(sentence)
                    if predicted == author:
                        correct += 1
                    else:
                        if self.print_misclassification:
                            print(f"Text: {' '.join(sentence)} : Predicted: {predicted} : Ground truth: {author}")
                print(f"{author}  {((correct/len(split['test'])) * 100):.2f}% correct")
        else:
            number = 0
            with open(self.testfile, 'r', encoding='utf-8') as file:
                for line in file:
                    sentence = word_tokenize(line.strip())
                    pred = self.apply_models(sentence)
                    if number == 0:
                        print(f"{pred} (the first line of text is predicted as {pred})")
                        number +=1
                    else:
                        print(pred)


    def generate_text(self, prompt, resp_length=50):

        for author, model in self.models.items():
            tokens = word_tokenize(prompt.strip())
            start_of_resp = len(tokens)
            context = tokens[-self.num_grams + 1:]

            resp = ""

            for i in range(resp_length):
                next_token = model.generate(1, text_seed=context)
                if next_token == '</s>':
                    print("predicted stop token")
                    break;
                # add to current tokens
                tokens.append(next_token)
                # update the context
                context = tokens[-self.num_grams + 1:]
                perplex_grams = list(ngrams(pad_both_ends(tokens, 2), self.num_grams))
            print(f"{author} model Generated: {' '.join(tokens)}")
            print(f"Perplexity of new generated text: {model.perplexity(perplex_grams):.2f}")


class DiscriminativeModel:
    """Auto model for sequence classification from hugging face"""
    def __init__(self, al, flag, testfile, model_name='distilbert-base-uncased', print_misclassification=False):
        self.test_flag = flag
        self.test_file = testfile
        self.model_name = model_name
        self.author_list = al
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.accuracy = evaluate.load('accuracy')
        self.print_misclassification = print_misclassification

        self.id2label = {}
        self.label2id = {}

        self.train_data = None
        self.test_data = None

        # initalize hardware devices
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        print(self.device)
        self.init_data()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.id2label), id2label=self.id2label, label2id=self.label2id)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        print("training LM... (this may take a while)")
        self.train_to_data()

    def init_data(self):
        """Creates the data for the model"""

        label = 0
        raw_data = []

        for filename in self.author_list:
            author = filename.split('_')[0]
            self.id2label[label] = author
            self.label2id[author] = label

            with open(filename, 'r', encoding='utf-8') as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        raw_data.append({'text': line, 'label': label})
                    label += 1

        random.shuffle(raw_data)
        # for i in range(30):
        #   print(raw_data[i])
        # print("made it to splitting")
        if not self.test_flag:
            # split
            index = int(len(raw_data) * .9)
            train_data = raw_data[:index]
            test_data = raw_data[index:]
        else:
            test_data = []
            with open(self.test_file, 'r', encoding='utf-8') as fp:
                for line in fp:
                    test_data.append({'text': line, 'label' : None})
        train_data = raw_data

        # print('made it past splitting')
        self.train_data = Dataset.from_dict({"text": [d['text'] for d in train_data], "label": [d['label'] for d in train_data]})
        self.test_data = Dataset.from_dict({"text": [d['text'] for d in test_data], "label": [d['label'] for d in test_data]})
        self.train_data = self.train_data.map(self.preprocess_function, batched=True)
        self.test_data = self.test_data.map(self.preprocess_function, batched=True)

    def preprocess_function(self,examples):
        return self.tokenizer(examples['text'], truncation=True)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.mean(predictions == labels)

        # Calculate per-class accuracy
        per_class_accuracy = {}
        for label_id, author in self.id2label.items():
            indices = labels == label_id
            class_predictions = predictions[indices]
            class_labels = labels[indices]
            class_accuracy = np.mean(class_predictions == class_labels)
            per_class_accuracy[author] = class_accuracy

        print("Per-author accuracy on the test set:")
        for author, acc in per_class_accuracy.items():
            print(f"{author}: {acc * 100:.2f}%")

        return {"accuracy": accuracy}

    def train_to_data(self):
        """ Function where the model gets trained"""
        start = time.time()
        lr = 2e-5
        epochs = 5

        train_args = TrainingArguments(
            output_dir='raysModel',
            learning_rate=lr,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy='no',
            save_strategy='no',
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_data,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        print(f"Training Time: {(time.time()-start):.2f} seconds")

    def evaluate(self):
        misclass = []
        if not self.test_flag:
            eval_args = TrainingArguments(output_dir='temp', per_device_eval_batch_size=32)
            eval_trainer = Trainer(
            model=self.model,
            args=eval_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            )
            results = eval_trainer.evaluate(eval_dataset=self.test_data)
            print(f"model accuracy: {results['eval_accuracy']:.2f}%")

            if self.print_misclassification:
                for item in self.test_data:
                    input = self.tokenizer(item['text'], truncation=True, padding=True, return_tensors="pt")
                    input.to(self.device)
                    outputs = self.model(**input)
                    predicted_label = torch.argmax(outputs.logits).item()
                    predicted_author = self.id2label[predicted_label]


                if predicted_label != item['label']:
                    print(f"{item['text']} : Predicted: {self.id2label[predicted_label]}, Ground truth: {predicted_author}")
        else:
            first = True
            predictions = []

            for item in self.test_data:
                input = self.tokenizer(item['text'], truncation=True, padding=True, return_tensors="pt")
                input.to(self.device)
                outputs = self.model(**input)
                predicted_label = torch.argmax(outputs.logits).item()
                predicted_author = self.id2label[predicted_label]
                if first is True:
                    print(f"{predicted_author} (the first line of text is predicted as {predicted_author})")
                    first = False
                else:
                    print(f"{predicted_author}")



if __name__ == "__main__":
    args = sys.argv
    args = args[1:]
    if 3 <= len(args) <= 5:
        testflag = False
        author_files = []
        testfile = None

        try:
            with open(args[0], 'r', encoding='utf-8') as file:
                for line in file:
                    author_files.append(line.strip('\n'))
        except FileNotFoundError as file_not_found:
            print(f"Could not open {args[0]}")
        except TypeError as tpe:
            print(f"File: {args[0]} is empty")

        if len(args) == 5:
            testflag = True
            testfile = args[-1]
            # print(testfile)

        if args[2] == "generative":
            print("Generative")
            prompt = "I should have objected very strongly "
            g1 = GenerativeModel(author_files, testflag, testfile=testfile, num_grams=2)
            g1.evaluate()

        else:
            print("Discriminative")
            d1 = DiscriminativeModel(author_files, testflag, testfile=testfile)
            d1.evaluate()