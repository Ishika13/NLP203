import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import wandb
import gc
from torch.cuda.amp import autocast, GradScaler

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Set PyTorch memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Initialize wandb or comment out if not needed
wandb.init(project="Seq2Seq_0")

# Hyperparameters - DRASTICALLY REDUCED
BATCH_SIZE = 16  # Reduced from 4
EMBEDDING_DIM = 128  # Reduced from 128
HIDDEN_DIM = 128  # Reduced from 128
MAX_SEQ_LENGTH = 256  # Hard limit on sequence length
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "tokenizer_1.pkl"
GRADIENT_ACCUMULATION_STEPS = 4  # Increased from 2

# Load or Initialize Tokenizer
if os.path.exists(TOKENIZER_PATH):
    print("Loading pre-saved tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
else:
    print("Initializing new tokenizer...")
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

# Tokenization Helper Function
def yield_tokens(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Truncate sequences before tokenization to save memory
            yield tokenizer(line.strip()[:MAX_SEQ_LENGTH*10])  # Approximate char to token ratio

# Load or Build Vocabulary
if os.path.exists("src_vocab_1.pth") and os.path.exists("tgt_vocab_1.pth"):
    print("Loading vocabularies from files...")
    src_vocab = torch.load("src_vocab_1.pth")
    tgt_vocab = torch.load("tgt_vocab_1.pth")
else:
    print("Building vocabulary...")
    src_vocab = build_vocab_from_iterator(yield_tokens("data/train.txt.src"), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
    tgt_vocab = build_vocab_from_iterator(yield_tokens("data/train.txt.tgt"), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
    src_vocab.set_default_index(src_vocab["<unk>"])
    tgt_vocab.set_default_index(tgt_vocab["<unk>"])
    torch.save(src_vocab, "src_vocab_1.pth")
    torch.save(tgt_vocab, "tgt_vocab_1.pth")

# Load Data with truncation
def load_data(src_path, tgt_path):
    print(f"Loading data from {src_path} and {tgt_path}...")
    src_texts, tgt_texts = [], []
    with open(src_path, 'r', encoding='utf-8') as f_src, open(tgt_path, 'r', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            # Pre-truncate to save memory during processing
            src_texts.append(src_line.strip()[:MAX_SEQ_LENGTH*10])  # Approximate char limit
            tgt_texts.append(tgt_line.strip()[:MAX_SEQ_LENGTH*10])  # Approximate char limit
    return src_texts, tgt_texts

# Dataset Class
class SummarizationDataset(data.Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        # Tokenize and convert to indices, with strict length limits
        src_tokens = tokenizer(self.src_texts[idx])[:MAX_SEQ_LENGTH]
        tgt_tokens = tokenizer(self.tgt_texts[idx])[:MAX_SEQ_LENGTH]
        
        src_tensor = torch.tensor([self.src_vocab[token] for token in src_tokens], dtype=torch.long)
        tgt_tensor = torch.tensor([self.tgt_vocab[token] for token in tgt_tokens], dtype=torch.long)
        
        return src_tensor, tgt_tensor

    def collate_fn(self, batch):
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        
        # Sort by length to minimize padding
        src_lengths = [len(x) for x in src_batch]
        sorted_indices = np.argsort(src_lengths)[::-1]
        
        src_batch = [src_batch[i] for i in sorted_indices]
        tgt_batch = [tgt_batch[i] for i in sorted_indices]
        
        # Apply padding
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.src_vocab["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_vocab["<pad>"])
        
        return src_batch, tgt_batch

# Load Training Data
print("Loading training data...")
train_src, train_tgt = load_data("data/train.txt.src", "data/train.txt.tgt")
dataset = SummarizationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
dataloader = data.DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=dataset.collate_fn, 
    num_workers=0,  # No multiprocessing to save memory
    pin_memory=False  # Disable pin_memory to reduce memory usage
)

# Fixed Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # Properly sized attention layer
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), attention_weights

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, output_dim)
        
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, embed_dim]
        context, _ = self.attention(hidden.squeeze(0), encoder_outputs)
        context = context.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, embed_dim + hidden_dim]
        
        # Get decoder outputs
        output, hidden = self.rnn(rnn_input, hidden)  # [batch_size, 1, hidden_dim], [1, batch_size, hidden_dim]
        
        # Combine output, embedded, and context for prediction
        output = torch.cat((output, embedded, context), dim=2)  # [batch_size, 1, hidden_dim*2 + embed_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        
        return prediction, hidden

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, tgt_len-1, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        decoder_input = tgt[:, 0]  # [batch_size]
        
        for t in range(1, tgt_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t-1] = decoder_output
            
            # Teacher forcing - next input is current target
            decoder_input = tgt[:, t]
            
        return outputs

# Initialize Model
print("Initializing model...")
attention = Attention(HIDDEN_DIM)
encoder = Encoder(len(src_vocab), EMBEDDING_DIM, HIDDEN_DIM)
decoder = Decoder(len(tgt_vocab), EMBEDDING_DIM, HIDDEN_DIM, attention)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
scaler = GradScaler()

def get_model_parameter_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

param_size = get_model_parameter_size(model)
print(f"Total number of parameters: {param_size}")

# Function to free up memory
def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Training Loop with memory optimizations
print("Training model...")
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    batch_count = 0
    
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader)):
        # Move data to device
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        # Use mixed precision training
        with autocast():
            # Forward pass
            output = model(src, tgt)
            
            # Calculate loss
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, target)
            loss = loss / GRADIENT_ACCUMULATION_STEPS  # Normalize for gradient accumulation
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Track loss
        epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        batch_count += 1
        
        # Update parameters after accumulating gradients
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
            # Unscale gradients and clip them
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Free memory
            free_memory()
            
            # Log to wandb
            wandb.log({"batch_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})
    
    # Print epoch stats
    avg_loss = epoch_loss / batch_count
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss})
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f"checkpoint_epoch_{epoch+1}.pt")
    
    # Free memory at end of epoch
    free_memory()

print("Training complete!")

# Save final model
torch.save(model.state_dict(), "seq2seq_final.pth")

# Fixed inference function
def generate_summary(model, input_text, max_length=50):
    model.eval()
    
    # Tokenize and convert to indices
    tokens = tokenizer(input_text)[:MAX_SEQ_LENGTH]
    src_tensor = torch.tensor([src_vocab[t] for t in tokens], dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # Get encoder outputs and hidden state
    encoder_outputs, hidden = model.encoder(src_tensor)
    
    # Start with <sos> token
    decoder_input = torch.tensor([tgt_vocab["<sos>"]], dtype=torch.long).to(DEVICE)
    
    # Store generated tokens
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get prediction from decoder
            decoder_output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
            
            # Get next token
            top_token = decoder_output.argmax(1)
            
            # Stop if <eos> token
            if top_token.item() == tgt_vocab["<eos>"]:
                break
                
            # Add token to generated tokens
            generated_tokens.append(top_token.item())
            
            # Update decoder input
            decoder_input = top_token
    
    # Convert token indices to words
    result = [tgt_vocab.get_itos()[t] for t in generated_tokens]
    
    # Remove special tokens
    result = [t for t in result if t not in ["<sos>", "<eos>", "<pad>", "<unk>"]]
    
    return " ".join(result)

# Generate predictions in batches to save memory
print("Generating predictions...")
with open("data/test.txt.src", 'r') as f_in, open("predicted_summaries.txt", 'w') as f_out:
    test_samples = f_in.readlines()
    batch_size = 16  
    
    for i in tqdm(range(0, len(test_samples), batch_size)):
        batch = test_samples[i:i+batch_size]
        
        for sample in batch:
            # Generate summary
            summary = generate_summary(model, sample.strip())
            f_out.write(summary + "\n")
        
        free_memory()