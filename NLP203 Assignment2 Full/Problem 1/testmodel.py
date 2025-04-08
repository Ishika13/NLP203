import json
import wikipediaapi
import allennlp_models
from allennlp.predictors.predictor import Predictor
import os

wiki_wiki = wikipediaapi.Wikipedia('en')

topics = ["Game of Thrones", "Youtube", "Dark matter", "Bermuda Triangle", "Apple"]

for i, topic in enumerate(topics, start=1):
    page = wiki_wiki.page(topic)
    
    if not page.exists():
        print(f"Topic '{topic}' not found on Wikipedia.")
        continue
    passage = ' '.join(page.summary.split()[:5000])  
    
    file_path = f"sample{i}.txt"
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(passage)
        print(f"Saved passage for '{topic}' in {file_path}")
    else:
        print(f"File '{file_path}' already exists and is not empty. Skipping saving.")
print("\nAll Wikipedia passages checked.\n")

predictor = Predictor.from_path("hf://allenai/transformer_qa")
questions = [
    "Who is the main character in Game of Thrones?",
    "What year did Game of Thrones begin filming?",
    "How many seasons were in Game of Thrones?",
    "What is the final episode of Game of Thrones?",
    "Who are the founders of YouTube?",
    "How much did Google pay for YouTube in 2006?",
    "When was YouTube founded?",
    "What is YouTube’s most popular video category?",
    "What is dark matter made of?",
    "What percentage of the universe’s mass is made up of dark matter?",
    "Is dark matter observable in the laboratory?",
    "How does dark matter interact with light?",
    "Where is the Bermuda Triangle located?",
    "What is the Bermuda Triangle’s nickname?",
    "Is the Bermuda Triangle considered dangerous by experts?",
    "Who discovered the first Apple?",
    "How many apple cultivars exist?",
    "Why are apple trees grafted onto rootstocks?",
    "What are the primary problems that apple trees face?",
    "What are apples commonly grown for?"
]
results = {}

for i in range(1, 6):  
    with open(f'sample{i}.txt', 'r', encoding="utf-8") as file:
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
    
    results[f'sample{i}.txt'] = question_answers

with open('results.json', 'w', encoding="utf-8") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved in 'results.json'")