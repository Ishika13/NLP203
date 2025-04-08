import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
import math
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import wandb
import gc
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Initialize wandb
wandb.init(project="Transformer_Seq2Seq")

# Hyperparameters optimized for transformer architecture
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
FFN_DIM = HIDDEN_DIM * 4  
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_ATTENTION_HEADS = 8
DROPOUT = 0.3
MAX_SEQ_LENGTH = 256
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001  
WARMUP_STEPS = 4000
CLIP_GRAD = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "tokenizer_transformer_1.pkl"
GRADIENT_ACCUMULATION_STEPS = 4

print(f"Using device: {DEVICE}")

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
            yield tokenizer(line.strip()[:MAX_SEQ_LENGTH*10])

# Load or Build Vocabulary
if os.path.exists("src_vocab_transformer_1.pth") and os.path.exists("tgt_vocab_transformer_1.pth"):
    print("Loading vocabularies from files...")
    src_vocab = torch.load("src_vocab_transformer_1.pth")
    tgt_vocab = torch.load("tgt_vocab_transformer_1.pth")
else:
    print("Building vocabulary...")
    src_vocab = build_vocab_from_iterator(yield_tokens("data/train.txt.src"), 
                                        specials=["<unk>", "<pad>", "<sos>", "<eos>"], 
                                        min_freq=3) 
    tgt_vocab = build_vocab_from_iterator(yield_tokens("data/train.txt.tgt"), 
                                        specials=["<unk>", "<pad>", "<sos>", "<eos>"], 
                                        min_freq=3)
    src_vocab.set_default_index(src_vocab["<unk>"])
    tgt_vocab.set_default_index(tgt_vocab["<unk>"])
    torch.save(src_vocab, "src_vocab_transformer.pth")
    torch.save(tgt_vocab, "tgt_vocab_transformer.pth")

print(f"Source vocabulary size: {len(src_vocab)}")
print(f"Target vocabulary size: {len(tgt_vocab)}")

PAD_IDX = src_vocab["<pad>"]
SOS_IDX = src_vocab["<sos>"]
EOS_IDX = src_vocab["<eos>"]

# Loading Data
def load_data(src_path, tgt_path):
    print(f"Loading data from {src_path} and {tgt_path}...")
    src_texts, tgt_texts = [], []
    with open(src_path, 'r', encoding='utf-8') as f_src, open(tgt_path, 'r', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_texts.append(src_line.strip()[:MAX_SEQ_LENGTH*10])
            tgt_texts.append(tgt_line.strip()[:MAX_SEQ_LENGTH*10])
    return src_texts, tgt_texts

# Dataset Class for Transformer
class TransformerDataset(data.Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_tokens = tokenizer(self.src_texts[idx])[:MAX_SEQ_LENGTH-2]
        tgt_tokens = tokenizer(self.tgt_texts[idx])[:MAX_SEQ_LENGTH-2]
        
        src_indices = [self.src_vocab[token] for token in src_tokens]
        tgt_indices = [self.tgt_vocab["<sos>"]] + [self.tgt_vocab[token] for token in tgt_tokens] + [self.tgt_vocab["<eos>"]]
        
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
        
        return src_tensor, tgt_tensor

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        
        for src, tgt in batch:
            src_batch.append(src)
            tgt_batch.append(tgt)
        
        # Apply padding
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.src_vocab["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_vocab["<pad>"])
        
        return src_batch, tgt_batch

# Load Training Data
print("Loading training data...")
train_src, train_tgt = load_data("data/train.txt.src", "data/train.txt.tgt")
dataset = TransformerDataset(train_src, train_tgt, src_vocab, tgt_vocab)
dataloader = data.DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=dataset.collate_fn, 
    num_workers=0,
    pin_memory=False
)

# Positional Encoding Layer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Multi-Head Attention Layer
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e4)
        
        attention = self.dropout(F.softmax(energy, dim=-1))
        
        x = torch.matmul(attention, V)
        
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)
        
        output = self.output_linear(x)
        
        return output, attention

# Position-wise Feed-Forward Layer
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()
        
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, ffn_dim, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ffn_norm(src + self.dropout(_src))
        
        return src

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, ffn_dim, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        _tgt, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.self_attn_norm(tgt + self.dropout(_tgt))
        _tgt, attention = self.cross_attention(tgt, enc_src, enc_src, src_mask)
        tgt = self.cross_attn_norm(tgt + self.dropout(_tgt))
        _tgt = self.positionwise_feedforward(tgt)
        tgt = self.ffn_norm(tgt + self.dropout(_tgt))
        
        return tgt, attention

# Full Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ffn_dim, dropout, max_length=5000):
        super(Encoder, self).__init__()
        
        self.token_embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_length)
        
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)
        
    def forward(self, src, src_mask):
        src = self.token_embedding(src) * self.scale
        src = self.positional_encoding(src)
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src

# Full Transformer Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads, ffn_dim, dropout, max_length=5000):
        super(Decoder, self).__init__()
        
        self.token_embedding = nn.Embedding(output_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_length)
        
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        tgt = self.token_embedding(tgt) * self.scale
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        
        attentions = []
        
        for layer in self.layers:
            tgt, attention = layer(tgt, enc_src, tgt_mask, src_mask)
            attentions.append(attention)
        
        output = self.fc_out(tgt)
        
        return output, attentions

# Complete Seq2Seq Transformer
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(tgt[:, :-1], enc_src, tgt_mask[:, :, :-1, :-1], src_mask)
        
        return output, attention

print("Initializing Transformer model...")
encoder = Encoder(
    input_dim=len(src_vocab), 
    hidden_dim=HIDDEN_DIM, 
    num_layers=NUM_ENCODER_LAYERS, 
    num_heads=NUM_ATTENTION_HEADS, 
    ffn_dim=FFN_DIM, 
    dropout=DROPOUT
)

decoder = Decoder(
    output_dim=len(tgt_vocab), 
    hidden_dim=HIDDEN_DIM, 
    num_layers=NUM_DECODER_LAYERS, 
    num_heads=NUM_ATTENTION_HEADS, 
    ffn_dim=FFN_DIM, 
    dropout=DROPOUT
)

model = Transformer(
    encoder=encoder, 
    decoder=decoder, 
    src_pad_idx=PAD_IDX, 
    tgt_pad_idx=PAD_IDX, 
    device=DEVICE
).to(DEVICE)

# Initialize parameters
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters")

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
        self.param_groups = self.optimizer.param_groups
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_lr(self):
        return self._rate
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

base_optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
optimizer = NoamOpt(
    model_size=HIDDEN_DIM,
    factor=2.0,
    warmup=WARMUP_STEPS,
    optimizer=base_optimizer
)

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
scaler = GradScaler()

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

print("Training Transformer model...")
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    batch_count = 0
    
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader)):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        with autocast():
            output, _ = model(src, tgt)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, target)
            loss = loss / GRADIENT_ACCUMULATION_STEPS  
        
        scaler.scale(loss).backward()
        
        epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        batch_count += 1
        
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            free_memory()
            
            wandb.log({
                "batch_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": optimizer.get_lr()
            })
    
    # Calculate epoch stats
    avg_loss = epoch_loss / batch_count
    
    # Record end time and calculate duration
    end_time.record()
    torch.cuda.synchronize()
    duration = start_time.elapsed_time(end_time) / 1000  
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Duration: {duration:.2f}s")
    
    wandb.log({
        "epoch": epoch+1, 
        "loss": avg_loss,
        "epoch_duration": duration
    })
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"New best model with loss: {best_loss:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  
            'loss': best_loss,
        }, f"best_transformer_1_model.pt")
    
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"transformer_1_checkpoint_epoch_{epoch+1}.pt")
    
    free_memory()

print("Training complete!")

torch.save(model.state_dict(), "transformer_1_final.pth")

def generate(self, src, max_length=100, beam_size=5):
    self.eval()
    
    src_mask = self.make_src_mask(src)
    
    with torch.no_grad():
        enc_src = self.encoder(src, src_mask)
        ys = torch.ones(src.size(0), 1).fill_(SOS_IDX).type_as(src).to(self.device)
        
        for i in range(max_length):
            tgt_mask = self.make_tgt_mask(ys)
            out, _ = self.decoder(ys, enc_src, tgt_mask, src_mask)
            pred = out[:, -1]
            next_word = pred.argmax(1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            if (next_word == EOS_IDX).all():
                break
                
        return ys

Transformer.generate = generate

# Function for generating a summary with beam search
def generate_summary(model, text, max_length=100, beam_size=5):
    model.eval()
    
    # Tokenize source text
    tokens = tokenizer(text)[:MAX_SEQ_LENGTH]
    src_tensor = torch.tensor([src_vocab[t] for t in tokens], dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # Generate summary
    with torch.no_grad():
        generated = model.generate(src_tensor, max_length)
    
    # Convert tokens to text, skipping special tokens
    summary_tokens = []
    for token in generated[0].cpu().numpy():
        if token in [SOS_IDX, EOS_IDX, PAD_IDX]:
            continue
        summary_tokens.append(tgt_vocab.get_itos()[token])
    
    summary = " ".join(summary_tokens)
    return summary

# Generate predictions
print("Generating predictions with transformer model...")
with open("data/test.txt.src", 'r') as f_in, open("predicted_summaries_transformer_1.txt", 'w') as f_out:
    test_samples = f_in.readlines()
    
    for i, sample in enumerate(tqdm(test_samples)):
        try:
            summary = generate_summary(model, sample.strip())
            f_out.write(summary + "\n")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            f_out.write("Error generating summary.\n")
        
        if (i + 1) % 50 == 0:
            free_memory()

print("All processing complete!")