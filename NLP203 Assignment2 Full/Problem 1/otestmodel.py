import json
from allennlp.predictors.predictor import Predictor

# Initialize the predictor for RoBERTa
predictor = Predictor.from_path("hf://allenai/transformer_qa")

# Define the questions
questions = [
    "Who is the main character?",
    "Who is harry?",
    "Who is the boy who lived?",
    "Who is a professor?",
    "Where do mr and mrs dursley live?",
    "What is a muggle?",
    "Who is a wizard?",
    "Who lives in Godric's Hollow?"
]

# Load the out-of-domain sample (osample1.txt)
with open('osample1.txt', 'r', encoding="utf-8") as file:
    passage = file.read().strip()

# Initialize a dictionary to store the answers
question_answers = {}

# Process the questions for the loaded passage
for question in questions:
    predictor_input = {
        "passage": passage,
        "question": question
    }
    
    # Get predictions from RoBERTa
    predictions = predictor.predict_json(predictor_input)
    answer = predictions['best_span_str']
    
    # Store the answer
    question_answers[question] = answer

# Save the results to a JSON file
with open('oresults.json', 'w', encoding="utf-8") as json_file:
    json.dump(question_answers, json_file, indent=4)

print("Results saved in 'oresults.json'")