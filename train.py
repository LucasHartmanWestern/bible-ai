# use PyTorch (https://pytorch.org)
import torch
import torch.nn as nn
from torch.nn import functional as F

# **** hyperparameters ****
batch_size = 64 # how many independent sequences to be processed in parallel
block_size = 256 # the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2 # 20% chance of randomly not being allowed to communicate

# optional seed
# torch.manual_seed(1337)

# read the .txt file
with open('bible-niv.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create sorted list of all characters that occur in .txt file
chars = sorted(list(set(text)))
# amount of characters in .txt file
vocab_size = len(chars)

# **** "tokenize" the vocab (create a mapping from characters to integers due to using character-level tokens) ****
# **** note this is not the optimal schema and is very simple (SentencePiece or tiktoken are potential alternative) ****

# create lookup table from character to integer
stoi = { ch:i for i, ch in enumerate(chars) }
# create lookup table from integer to character
itos = { i:ch for i, ch in enumerate(chars) }

# takes a string and outputs a list of integers
encode = lambda s: [stoi[c] for c in s]
# takes a list of integers and outputs a string
decode = lambda l: ''.join([itos[i] for i in l])

# store text encoding into tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split data into training and validation sets
# the first 90% will be training data, and the rest validation data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# generate a small batch of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == 'train' else val_data # set either training or validation data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get random offset of data set
    x = torch.stack([data[i:i+block_size] for i in ix]) # get context (input)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # get target
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create one head of self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # create lower triangular matrix of ones which is TxT
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores
        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        # make upper triangular matrix of -infinity
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # use softmax to normalize matrix
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        # get matrix of averages on matrix x
        v = self.value(x) # (B, T, C)
        out = weights @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

# have multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# a simple linear layer followed by non-linearity
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout) # randomly prevent nodes from communicating to prevent overfitting
        )

    def forward(self, x):
        return self.net(x)

# transformer block: communication followed by computation
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # handles the communication
        x = x + self.ffwd(self.ln2(x)) # handles the computation
        return x

# feed the batch of data into a neural network using the Bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # create a token embedding table of size (vocab_size x vocab_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both a (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reorder to work with cross_entropy built-in functionality
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # measure quality of logits with respect to the targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

# instantiate BigramLanguageModel
model = BigramLanguageModel()
m = model.to(device)

# optimize model so that it doesn't just generate random strings
# this is the step where the Bigram model is actually being trained

if __name__ == "__main__": # don't train model when simply generating

    # create a PyTorch optimizer (using AdamW)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True) # zero out gradients from previous step
        loss.backward() # get gradients for all parameters
        optimizer.step() # use gradients to update parameters

    # Save the trained model
    model_save_path = "bigram_language_model.pt"
    torch.save(model.state_dict(), model_save_path)