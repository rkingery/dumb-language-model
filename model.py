import torch
from torch import nn
import torch.nn.functional as F

device = 'cpu'

class ModifiedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, 
                 batch_first=True, norm_first=False, device=device, dtype=None, alpha=1, beta=1):
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **kwargs)
        self.norm_first = norm_first if not (alpha == 1 and beta == 1) else False # deepnorm reqs post-norm
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        
        self.alpha = alpha # NEW
        self.beta = beta # NEW

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            x = self.alpha * x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask) # UPDATED
            x = self.alpha * x + self._ff_block(self.norm2(x)) # UPDATED
        else:
            x = self.norm1(self.alpha * x + self._sa_block(x, src_mask, src_key_padding_mask)) # UPDATED
            x = self.norm2(self.alpha * x + self._ff_block(x)) # UPDATED
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        q, k, v = x / self.beta, x / self.beta, x # NEW
        x = self.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000, N=10_000, dropout=0.1):
        super().__init__()
        position = torch.arange(max_len)[:, None]
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-torch.log(torch.tensor(N) / emb_size)))
        self.encoding = torch.zeros(max_len, 1, emb_size)
        self.encoding[:, 0, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x.shape = y.shape = (batch_size, seq_len, emb_size)
        batch_size = x.shape[0]
        p = self.encoding[:batch_size].to(device)
        x += p
        y = self.dropout(x)
        return y
    
class EncoderLM(nn.Module):
    def __init__(self, vocab_size, emb_size, num_layers=3, nhead=8, dropout=0.1, dim_feedforward=256,
                 masking=False, max_len=1024, padding_idx=None, deepnorm=False, activation=nn.GELU(),
                 norm_first=False):
        super().__init__()
        self.masking = masking
        self.padding_idx = padding_idx
        self.emb_size = emb_size
        self.alpha = (2 * num_layers)**(1/4) if deepnorm else 1
        self.beta = (8 * num_layers)**(-1/4) if deepnorm else 1
        self.mask = torch.triu(torch.ones(max_len, max_len)*(-torch.inf), diagonal=1).to(device)
        self.positional = PositionalEncoding(emb_size, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.transformer = nn.TransformerEncoder(ModifiedTransformerEncoderLayer(
            emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, 
            alpha=self.alpha, beta=self.beta, activation=activation, norm_first=norm_first), num_layers)
        self.fc = nn.Linear(emb_size, vocab_size)
        self.apply(self.init_weights)

    def forward(self, X):
        # X.shape = (batch_size, seq_len)
        # Yhat.shape = (batch_size, seq_len, vocab_size)
        batch_size, seq_len = X.shape
        mask = self.mask[:seq_len, :seq_len] if self.masking else None
        pad_mask = (X == self.padding_idx) if self.padding_idx is not None else None
        X = self.embedding(X) * torch.sqrt(torch.tensor(self.emb_size))
        X = self.positional(X)
        X = self.transformer(X, mask, pad_mask)
        X = self.fc(X)
        return X
    
    def init_weights(self, layer):
        if isinstance(layer, nn.modules.linear.Linear) or \
           isinstance(layer, nn.modules.sparse.Embedding) or \
           isinstance(layer, nn.modules.linear.NonDynamicallyQuantizableLinear):
            torch.nn.init.xavier_normal_(layer.weight, gain=self.beta)
            
class Seq2SeqCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, Yhat, Y):
        bs, bptt = Y.shape[0], Y.shape[1]
        y = Y.reshape(bs * bptt,)
        yhat = Yhat.reshape(bs * bptt, -1)
        loss = self.loss_fn(yhat, y).mean()
        return loss
    
    
if __name__ == '__main__':
    model = EncoderLM(10_000, 512, num_layers=1, nhead=8, dim_feedforward=512,
              masking=True, padding_idx=1, dropout=0.1, max_len=525, deepnorm=True).to(device)
    print(model)