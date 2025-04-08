import json
from allennlp.predictors.predictor import Predictor
import os

sample_files = [f"sample{i}.txt" for i in range(1, 6)]

predictor = Predictor.from_path("hf://allenai/transformer_qa")

questions = [
    "Who is the legendary mastermind behind Game of Thrones?",
    "Who is the main character in Game of Thrones?",
    "What year did Game of Thrones begin filming?",
    "How many seasons were in Game of Thrones?",
    "What is the final episode of Game of Thrones?",
    "Who are the legendary masterminds behind YouTube?",
    "How much did Google splash out for YouTube in 2006?",
    "How much did Google pay for YouTube in 2006?",
    "When did YouTube magically appear on the internet?",
    "What is YouTube’s top-tier video category, aka the undisputed champion?"
]

results = {}

for file_path in sample_files:
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        print(f"File '{file_path}' is missing or empty. Skipping...")
        continue

    with open(file_path, 'r', encoding="utf-8") as file:
        passage = file.read().strip()
    
    question_answers = {}

    for question in questions:
        predictor_input = {
            "passage": passage,
            "question": question
        }

        predictions = predictor.predict_json(predictor_input)
        answer = predictions['best_span_str']
        question_answers[question] = answer
    
    results[file_path] = question_answers

with open('mresults.json', 'w', encoding="utf-8") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved in 'mresults.json'")