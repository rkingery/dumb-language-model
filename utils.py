import torch
import torchtext
from pathlib import Path

from model import EncoderLM

MAX_LEN = 50
TEMPERATURE = 0.5

device = 'cpu'
model_dir = Path().cwd() / 'models'
model_path = model_dir / 'lm_8_layers.pth'
vocab_path = model_dir / 'vocab_8_layers.pth'

emb_size = 768
dim_feedforward = 2048
num_layers = 8
nhead = 12

def get_vocab():
    return torch.load(vocab_path, map_location=torch.device(device))

vocab = get_vocab()
vocab_size = len(vocab)
padding_idx = vocab.get_stoi()['<pad>']

def get_model():
    model = EncoderLM(vocab_size, emb_size, num_layers=num_layers, nhead=8, dim_feedforward=dim_feedforward,
                  masking=True, padding_idx=padding_idx, dropout=0.1, max_len=525, deepnorm=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model

def clean_text(tokens):
    text = []
    prev_token = '<bos>'
    for token in tokens:
        if token != '<unk>':
            if prev_token == '<up>':
                token = token.upper()
            if prev_token == '<cap>':
                token = token.title()
        if token == '@-@':
            token = '-'
        if token not in ['<bos>', '<eos>', '<up>', '<cap>']:
            text.append(token)
        prev_token = token
    return ' '.join(text)

def generate_text(seed, model, vocab, max_len=20, temperature=0.5, device=device, skip_tokens=['<unk>']):
    stoi, itos = vocab.get_stoi(), vocab.get_itos()
    stoi_map = lambda word: stoi[word] if word in stoi.keys() else stoi['<unk>']
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    model = model.eval()
    seed_tokens = ['<bos>'] + tokenizer(seed)
    x = torch.tensor([stoi_map(word) for word in seed_tokens]).long().to(device)[None, :]
    idxs = []
    probs = []
    idx_prev = stoi['<unk>']
    for _ in range(max_len):
        yhat = model(x)
        prob = yhat[:, -1].softmax(dim=-1)
        if (torch.rand(1) < temperature) or (idx_prev in [stoi_map(token) for token in skip_tokens]):
            idx = torch.multinomial(prob, 1, replacement=True).item()
        else:
            idx = prob.argmax(-1).item()
        idxs.append(idx)
        probs.append(prob.unsqueeze(dim=1))
        x = torch.cat([x, torch.ones(1, 1).fill_(idx).long().to(device)], dim=1)
        idx_prev = idx
        if itos[idx] == '<eos>':
            break
    probs = torch.cat(probs, dim=1)
    generated = [itos[idx] for idx in idxs]
    text = seed + ' ' + clean_text(generated)
    return text


if __name__ == '__main__':
    vocab = get_vocab(vocab_path)
    model = get_model(model_path)
    seed = 'The entropy of the universe is'
    generated = generate_text(seed, model, vocab, max_len=20, temperature=0.5, device=device, skip_tokens=['<unk>'])
    print(generated)