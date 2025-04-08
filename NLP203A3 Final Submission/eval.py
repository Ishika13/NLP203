from rouge import Rouge

# File paths
hyp_path = '/home/ikulkar1/Assignments/NLP203/A3/predicted_summaries_transformer.txt'
ref_path = '/home/ikulkar1/Assignments/NLP203/A3/data/test.txt.tgt'

# Read files
with open(hyp_path, 'r') as f:
    hypothesis = [line.strip() for line in f.readlines()]

with open(ref_path, 'r') as f:
    reference = [line.strip() for line in f.readlines()]

# Ensure same length by filtering out missing predictions
filtered_hypothesis = []
filtered_reference = []

for hyp, ref in zip(hypothesis, reference):
    if hyp:  # Only keep pairs where hypothesis exists
        filtered_hypothesis.append(hyp)
        filtered_reference.append(ref)

# Calculate ROUGE
rouge = Rouge()
scores = rouge.get_scores(filtered_hypothesis, filtered_reference, avg=True)
print("Filtered ROUGE Scores:", scores)