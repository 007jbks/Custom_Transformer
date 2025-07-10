import numpy as np
text = open('pg48320.txt','r').read()
vocab = list(set(sorted(text)))
vocab_len = len(vocab)
# Now we need to create the embedding matrix of size vocab_len * d_model
char_to_idx = {c:i for i,c in enumerate(vocab)}  # this is a lookup table
idx_to_char = {i:c for i,c in enumerate(vocab)}

# embeddign matrix
d_model = 768
embedding_matrix = np.random.randn(vocab_len,d_model)
input_string =  "My name is Sherlock Holmes and I"
X = []
for i in input_string:
  X.append(embedding_matrix[char_to_idx[i]])
X = np.array(X)

def get_positional_encoding(seq_len, d_model):   # this is actually based on the actual transformer paper
    pos = np.arange(seq_len)[:, np.newaxis]               # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]                 # (1, d_model)
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / d_model)
    angle_rads = pos * angle_rates                        # (seq_len, d_model)

    # Apply sin to even indices, cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads
pos_encoding = get_positional_encoding(len(X), d_model)
X += pos_encoding
## Now comes the attention part the encoding is over now
d_k = d_model # one head for now
Q = np.random.randn(d_model,d_k)
K = np.random.randn(d_model,d_k)
V = np.random.randn(d_model,d_k)
Q = (Q @ X.T).T  # Shape: (32, 768)
K = (K @ X.T).T
V = (V @ X.T).T
# let's calculate attention then
K_T = K.T
Q_K_T = np.matmul(Q,K_T)

Q_K_T /= np.sqrt(d_k)

def softmax(x,axis=1):
  x = x-np.max(x,axis=axis,keepdims=True)
  exp_x = np.exp(x)
  return exp_x/np.sum(exp_x,axis=axis,keepdims=True)

attention_output = np.dot(softmax(Q_K_T),V)
A_X = attention_output
def layernorm(X, epsilon=1e-5):
    # X: shape (seq_len, d_model)
    mean = np.mean(X, axis=-1, keepdims=True)  # (seq_len, 1)
    var = np.var(X, axis=-1, keepdims=True)    # (seq_len, 1)
    normalized = (X - mean) / np.sqrt(var + epsilon)

    # Optional: learnable scale and bias (initialize as 1 and 0 for now)
    gamma = np.ones_like(X)
    beta = np.zeros_like(X)
    
    return gamma * normalized + beta
X = layernorm(X+A_X)
def ReLU(x):
  return np.maximum(0,x)
d_ff = 4 * d_model  # 3072
# Initialize weights
W1 = np.random.randn(d_model, d_ff) * 0.01  # (768, 3072)
b1 = np.zeros((1, d_ff))

W2 = np.random.randn(d_ff, d_model) * 0.01  # (3072, 768)
b2 = np.zeros((1, d_model))

def feedforward(X):
    return ReLU(X @ W1 + b1) @ W2 + b2  # Shape: (seq_len, d_model)
ffn_out = feedforward(X)  # X: (seq_len, d_model)
X = layernorm(X + ffn_out)  # Add & Norm
