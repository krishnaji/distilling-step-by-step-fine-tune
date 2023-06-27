from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, Seq2SeqTrainingArguments,DataCollatorForSeq2Seq
from datasets import load_dataset , load_metric
import numpy as np
import glob
import logging
import os
import nltk
import evaluate

from google.cloud import storage

# 1. Initialize the model and tokenizer
model_name = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,use_cache=False)

# 2. Load the datasets
datasets = load_dataset('json', data_files='dataset.jsonl')
train_dataset = datasets['train'].train_test_split(test_size=0.2)['train']
eval_dataset = datasets['train'].train_test_split(test_size=0.2)['test']

# Initialize the metric
rouge_score = evaluate.load('rouge')


# Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples['input_text'], padding='max_length', truncation=True, max_length=512)
    outputs = tokenizer(examples['output_text'], padding='max_length', truncation=True, max_length=512)
    outputs["input_ids"] = [list(map(lambda id: id if id != 0 else -100, label)) for label in outputs["input_ids"]]
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': outputs['input_ids']}

# Map the tokenization function to the training and evaluation datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set the format of the datasets to output torch.Tensor
train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

nltk.download("punkt")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = [
        '\n'.join(sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        '\n'.join(sent_tokenize(label.strip())) for label in decoded_labels
    ]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

# 3. Setup the training configurations
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir= os.environ["AIP_TENSORBOARD_LOG_DIR"],
    report_to=["tensorboard"], 
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    deepspeed="configs/ds_flan_t5_base.json",
    fp16=False,
    gradient_checkpointing=True,
    load_best_model_at_end=False,
)

# 4. Create the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# Train the model
trainer.train()
trainer.evaluate()
# Save the model
tokenizer.save_pretrained(f'model_tokenizer')
trainer.save_model('model_results')

# Save model to gcs
output_directory = os.environ['AIP_MODEL_DIR']

bucket_name = output_directory.split("/")[2] 
object_name = "/".join(output_directory.split("/")[3:])

directory_path = "model_results"
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)

directory_path = "model_tokenizer"  

client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)