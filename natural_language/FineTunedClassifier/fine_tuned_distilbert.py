# -*- coding: utf-8 -*-
"""
Fine tuning a model using dataset from Hugging Face

"""
import torch
import math
import time
from tqdm import tqdm
import wandb
import matplotlib as mtpl
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline

custom_train = True

if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'
print(f'device: {device}')

train_ds = load_dataset('yaful/DeepfakeTextDetect', split="train")
test_ds = load_dataset('yaful/DeepfakeTextDetect', split="test")

print(f"Length of train_ds: {len(train_ds)}")
print(f"Length of test_ds: {len(test_ds)}")

# Set the shuffle seed
my_seed = 19

# Shuffle data
train_ds = train_ds.shuffle(seed=my_seed)
test_ds = test_ds.shuffle(seed=my_seed)

# Lengths of the data
print(f"Number of training samples computer written: {len(train_ds.filter(lambda count: count['label'] == 0))}")
print(f"Number of trainng samples human written: {len(train_ds.filter(lambda count: count['label'] == 1))}\n")

# Check split between labels
print(f"Number of test samples computer written: {len(test_ds.filter(lambda count: count['label'] == 0))}")
print(f"Number of test samples human written: {len(test_ds.filter(lambda count: count['label'] == 1))}")

# Hugging face model
my_model_name = 'distilbert-base-uncased'

# Text tokenizer
my_tokenizer = AutoTokenizer.from_pretrained(my_model_name)

# Preprocess text
def preprocess_function(examples):
    """
    Tokenizes the data
    """
    return my_tokenizer(examples['text'], truncation=True)


# Tokenize the datasets
tokenized_train_ds = train_ds.map(preprocess_function, batched=True)
tokenized_test_ds = test_ds.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=my_tokenizer)

accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    """
    Computes the metrics while training
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Initialize the labels for the data
# Makes them human readable
labels = ['COMPUTER', 'HUMAN']
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in id2label.items()}

# Ensures labels are set
print('id2label:', id2label)
print('label2id:', label2id)

# Load the model from hugging face
model = AutoModelForSequenceClassification.from_pretrained(my_model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

# Training Parameters
epochs = 8
lr = 2e-6

train_args = TrainingArguments(
    output_dir='ray_rays_model',
    learning_rate=lr,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Init weights and biases for training tracking
wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="ray_rays_project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "dataset" : "yaful/DeepfakeTextDetect",
    "architecture": "distilbert-base-uncased",
    "epochs": epochs,
    }
)


# Default Training class given by hugging face
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_test_ds,
    tokenizer=my_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if custom_train is False:
    trainer.train()



# Custom trainer class to finetune the training parameters
class MyTrainer(Trainer):
  def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
    number_of_epochs = args.num_train_epochs
    start = time.time()
    train_loss = []
    train_acc = []
    eval_acc = []

    criterion = torch.nn.CrossEntropyLoss().to(device)
    self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

    train_dataloader = self.get_train_dataloader()
    eval_dataloader = self.get_eval_dataloader()

    max_steps = math.ceil(args.num_train_epochs * len(train_dataloader))

    for epoch in range(number_of_epochs):
      train_loss_per_epoch = 0
      train_acc_per_epoch = 0
      with tqdm(train_dataloader, unit="batch") as training_epoch:
        training_epoch.set_description(f"Training Epoch {epoch + 1}")
        for step, inputs in enumerate(training_epoch):
          inputs = inputs.to(device)
          labels = inputs['labels']

          # forward pass
          self.optimizer.zero_grad()

          output = self.model(**inputs)

          # get the loss
          loss = criterion(output['logits'], labels)

          train_loss_per_epoch += loss.item()

          # calculate gradients
          loss.backward()
          #update the weights
          self.optimizer.step()
          train_acc_per_epoch += (output['logits'].argmax(1) == labels).sum().item()

        # Adjust teh learnng rate
        self.scheduler.step()
        train_loss_per_epoch /= len(train_dataloader)
        train_acc_per_epoch /= (len(train_dataloader) * batch_size)

        eval_loss_per_epoch = 0
        eval_acc_per_epoch = 0

        with tqdm(eval_dataloader, unit="batch") as eval_epoch:
          # similar to the training loop that does this

          eval_epoch.set_description(f"Evaluation Epoch {epoch + 1}")

          for eval_step, eval_input in enumerate(eval_epoch):
            eval_inputs = eval_input.to(device)
            labels = eval_inputs['labels']

            # Outputs
            output = self.model(**eval_inputs)

            loss = criterion(output['logits'], labels)

            eval_loss_per_epoch += loss.item()
            # no need for back propogation

            eval_acc_per_epoch += (output['logits'].argmax(1) == labels).sum().item()

        eval_loss_per_epoch /= (len(eval_dataloader))
        eval_acc_per_epoch /= (len(eval_dataloader) * batch_size)

        print(f"\tTrain Loss: {train_loss_per_epoch: .3f} | Train Acc: {train_acc_per_epoch * 100: .2f}%")
        print(f"\tEval Loss: {eval_loss_per_epoch: .3f} | Eval Acc: {eval_acc_per_epoch * 100: .2f}%")

        wandb.log({"Train Loss": train_loss_per_epoch, "Train Acc": train_acc_per_epoch, "Eval Loss": eval_loss_per_epoch, "Eval Acc": eval_acc_per_epoch})

      print(f"Time: {(time.time() - start) / 60: .3f} minutes\n")


# Create new Trainer
my_train = MyTrainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_test_ds,
    tokenizer=my_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)

# Train model
my_train.train()
wandb.finish()

# Save the model locally
my_train.save_model("./my_model")

# Push the model to hugging face for saving
model.push_to_hub('lyon0210/fine-tuned-model-ray')

# Load the model once saved
path = './my_model'
model = AutoModelForSequenceClassification.from_pretrained(path)

# Applying the model
text_to_classify = ""

classifier = pipeline('sentiment-analysis', model = model, tokenizer=my_tokenizer, device=device)

inputs = my_tokenizer(text_to_classify, return_tensors='pt').to(device)

with torch.no_grad():
  logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(logits.argmax().item())
model.config.id2label[predicted_class_id]
