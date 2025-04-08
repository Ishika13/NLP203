import json
import torch
from transformers import pipeline

# Loading the model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def load_covid_qa(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]

def generate_predictions(dataset, output_file):
    print(f"Generating predictions for {len(dataset)} articles")
    predictions = {}

    for article in dataset:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question_id = qa["id"]
                question = qa["question"]
                result = qa_pipeline(question=question, context=context)    
                predictions[question_id] = result["answer"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
    
    print(f"Predictions saved to {output_file}")

# Loading the data
dev_data = load_covid_qa("covid-qa/covid-qa-dev.json")
test_data = load_covid_qa("covid-qa/covid-qa-test.json")

# Generating predictions
generate_predictions(dev_data, "predictions_dev.json")
generate_predictions(test_data, "predictions_test.json")

print("Inference complete.")