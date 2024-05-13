# Source: https://www.philschmid.de/fine-tune-flan-t5
# tensorboard: tensorboard --logdir log
# Run the following in terminal before python3 FbFTL_LLM/test.py:  huggingface-cli login        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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torchinfo import summary
from datasets import load_dataset
from random import randrange
from datasets import concatenate_datasets
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import sys
from collections import deque

HfFolder.save_token('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')

# FL parameters
FL_type = 'FbFTL'  # 'FL', 'FTLf'(same as FL), 'FTLc', 'FbFTL'
model_to_train = [False, 4, 0, True]  # available for 'FTLc' and 'FbFTL': [False, 4, 0, True]
# shared: (boolean), encoder: (int:0~8, index after which is trained, 10 to not train layer norm), decoder: (same as encoder), lm_head: (boolean)
train_set_denominator = -1  # -1 to use full training dataset, or 0 < int <= 14732 : pick a subset with [int] training samples
# Hyperparameters
NUM_CLASSES = 32128
learning_rate = 2e-4  # TODO: source code 5e-5, full model 2e-4
num_train_epochs = 20  # TODO: source code 5, full model 20
batch_size = 8  # source code 8
# TODO: implement following params
# if FL_type == 'FL':
#     transfer, full = False, True  # transfer or train model from scratch   # train whole model or last few layers
#     sigma = 0.  # 0.5 relative std for addtive gaussian noise on gradients
# elif FL_type == 'FTLf':
#     transfer, full = True, True  
#     sigma = 0.  # 0.3305 relative std for addtive gaussian noise on gradients
# elif FL_type == 'FTLc':
#     transfer, full = True, False 
#     sigma = 0.  # 0.285 relative std for addtive gaussian noise on gradients
# elif FL_type == 'FbFTL':
#     transfer, full = True, False 
#     sigma = 0  # 0.8? relative std for addtive gaussian noise on features
#     saved_noise = True  # save noise at beginning
# else:
#     raise ValueError('Unknown FL_type: ' + FL_type)
# relative_noise_type = 'all_std'  # 'individual', 'all_std'
# packet_loss_rate = 0.  # 0, 0.05, 0.1, 0.15
# quan_digit = 32  # digits kept after feature quantization: None (max:(12~18)(6~8), min=0, std~0.8) or int
# sparse_rate = 0.9  # ratio of uplink elements kept after sparsification: None or (0,1]

# Load dataset from the hub
dataset_id = "samsum"
dataset = load_dataset(dataset_id)
if train_set_denominator != -1:
    dataset['train'] = dataset['train'].select(range(train_set_denominator))
print(f"Train dataset size: {len(dataset['train'])}")  # 14732
print(f"Test dataset size: {len(dataset['test'])}")  # 819
sample = dataset['train'][randrange(len(dataset["train"]))]
# print(f"dialogue: \n{sample['dialogue']}\n---------------")
# print(f"summary: \n{sample['summary']}\n---------------")
# sys.exit()

# Loading the Model
model_id="google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
# for parameter in model.parameters():  # model: shared, encoder, decoder, lm_head
#     parameter.requires_grad = False
if FL_type in ['FTLc', 'FbFTL'] and not model_to_train[0]:
    for parameter in model.shared.parameters():  # initial embedding layer, not trained
        parameter.requires_grad = False

if FL_type in ['FTLc', 'FbFTL']:
    for i, m in enumerate(model.encoder.block):  # whether train encoder blocks
        if i < model_to_train[1]:
            for parameter in m.parameters():
                parameter.requires_grad = False
    if model_to_train[1] >= 10:
        for parameter in model.encoder.final_layer_norm.parameters():  # whether train encoder layer norm
            parameter.requires_grad = False

if FL_type in ['FTLc', 'FbFTL']:
    for i, m in enumerate(model.decoder.block):  # whether train decoder blocks
        if i < model_to_train[2]:
            for parameter in m.parameters():
                parameter.requires_grad = False
    if model_to_train[2] >= 10:
        for parameter in model.decoder.final_layer_norm.parameters():  # whether train decoder layer norm
            parameter.requires_grad = False

if FL_type in ['FTLc', 'FbFTL'] and not model_to_train[3]:
    for parameter in model.lm_head.parameters():  # final output layer, always trained
        parameter.requires_grad = False
summary(model)
# print(model)
# for param in model.state_dict():
#     print(param)
# sys.exit()

# Preprocess data
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), 
                                                                                 batched=True, remove_columns=["dialogue", "summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), 
                                                                                  batched=True, remove_columns=["dialogue", "summary"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]
    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# Metric
metric = evaluate.load("rouge")
# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# Hugging Face repository id
repository_id = f"{model_id.split('/')[1]}-{dataset_id}"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

trainer.evaluate()

# Save our tokenizer and create model card
tokenizer.save_pretrained(repository_id)
trainer.create_model_card()
# Push the results to the hub
# trainer.push_to_hub()


# Run Inference and summarize ChatGPT dialogues
# from transformers import pipeline
# from random import randrange
# # load model and tokenizer from huggingface hub with pipeline
# summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum", device=0)
# # select a random test sample
# sample = dataset['test'][randrange(len(dataset["test"]))]
# print(f"dialogue: \n{sample['dialogue']}\n---------------")
# # summarize dialogue
# res = summarizer(sample["dialogue"])
# print(f"flan-t5-base summary:\n{res[0]['summary_text']}")