from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from sklearn.metrics import f1_score
import numpy as np

# Loading the model and tokenizer
model_checkpoint = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Loading the dataset
data_files = {
    "train": "/home/ikulkar1/Assignments/NLP203/A2/Assignment2V1/Problem 2/covid-qa/covid-qa-train.json",
    "validation": "/home/ikulkar1/Assignments/NLP203/A2/Assignment2V1/Problem 2/covid-qa/covid-qa-dev.json",
    "test": "/home/ikulkar1/Assignments/NLP203/A2/Assignment2V1/Problem 2/covid-qa/covid-qa-test.json"
}
dataset = load_dataset("json", data_files=data_files, field="data")

# Preprocessing the data
def preprocess_function(examples):
    questions = []
    contexts = []
    answers = []

    paragraph = examples["paragraphs"]
    
    for para in paragraph:
        for p in para:
            for qa in p["qas"]:
                questions.append(qa["question"])
                contexts.append(p["context"])
                answers.append(qa["answers"])

    inputs = tokenizer(questions, contexts, truncation=True, padding="max_length", max_length=384)

    start_positions = [ans[0]["answer_start"] for ans in answers]  
    end_positions = [ans[0]["answer_start"] + len(ans[0]["text"]) for ans in answers]

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["paragraphs"])

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# Defining the training arguments
training_args = TrainingArguments(
    max_grad_norm=1.0,
    output_dir="./roberta-covid-qa",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    fp16=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred  
    predictions = np.argmax(logits, axis=1)  
    f1 = f1_score(labels, predictions, average="weighted") 
    return {"eval_f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Training the model
trainer.train()
metrics = trainer.evaluate()
print(metrics)
print(trainer.state.log_history)

# Saving the model
model.save_pretrained("./roberta-covid-qa")
tokenizer.save_pretrained("./roberta-covid-qa")