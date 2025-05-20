import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import os
import wandb
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Change 'te' → desired language
BASE_DIR = '/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/te/lexicons'


class CharacterEmbedding(nn.Module):
    # Creating an embedding layer that maps input character indices to embedding vectors.
    # input_size: number of unique characters (vocabulary size)
    # embedding_dim: size of each embedding vector
    def __init__(self, input_size, embedding_dim):
        super(CharacterEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)

    # Returns corresponding embedding vectors of shape (batch_size, seq_length, embedding_dim)
    def forward(self, input_seq):
        # input_seq: a tensor of character indices, typically of shape (batch_size, seq_length)
        return self.embedding(input_seq)


# EncoderRNN transforms sequences of token IDs into contextual hidden states
# Supports GRU, LSTM, or vanilla RNN cells
# input_size: number of unique tokens
# hidden_size: size of the RNN hidden state
# embedding_dim: size of token embedding vectors
# num_layers: number of stacked recurrent layers
# cell_type: 'GRU', 'LSTM', or 'RNN'
# dropout_p: dropout probability between RNN layers (only if num_layers > 1)
# bidirectional: whether to run the RNN in both forward and backward directions
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers=1, cell_type='GRU', dropout_p=0.1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim)

        # Dropout before the RNN (applied to embeddings)
        self.dropout = nn.Dropout(dropout_p)
        dropout_p = dropout_p if num_layers > 1 else 0

        # RNN layer
        if cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers,
                              dropout=dropout_p, bidirectional=bidirectional, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers,
                               dropout=dropout_p, bidirectional=bidirectional, batch_first=True)
        else:  # Default to RNN
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, dropout=dropout_p,
                              bidirectional=bidirectional, nonlinearity='tanh', batch_first=True)

    # Forward pass through the encoder
    def forward(self, input_seq):
        # Input shape: [batch_size, seq_len]
        batch_size = input_seq.size(0)

        # Convert indices to embeddings and apply dropout to embeddings
        # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)

        # Pass through RNN
        outputs, hidden = self.rnn(embedded)

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Repeat hidden state seq_len times to concat with encoder outputs
        # [batch_size, seq_len, hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # [batch_size, seq_len, hidden_size]
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)  # [batch_size, hidden_size, seq_len]

        v = self.v.repeat(batch_size, 1).unsqueeze(
            1)  # [batch_size, 1, hidden_size]

        energy = torch.bmm(v, energy).squeeze(1)  # [batch_size, seq_len]

        attn_weights = F.softmax(energy, dim=1)  # [batch_size, seq_len]

        return attn_weights


class DecoderRNNWithAttention(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_dim, num_layers=1,
                 cell_type='GRU', dropout_p=0.1):

        super(DecoderRNNWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # Embedding layer for target characters
        self.embedding = nn.Embedding(output_size, embedding_dim)

        # Dropout applied before RNN
        self.dropout = nn.Dropout(dropout_p)

        dropout_p = dropout_p if num_layers > 1 else 0

        # RNN input size is embedding_dim + hidden_size (due to attention context)
        rnn_input_size = embedding_dim + hidden_size

        # RNN layer
        if cell_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_size, hidden_size,
                              num_layers, dropout=dropout_p, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_size, hidden_size,
                               num_layers, dropout=dropout_p, batch_first=True)
        else:
            self.rnn = nn.RNN(rnn_input_size, hidden_size, num_layers,
                              dropout=dropout_p, nonlinearity='tanh', batch_first=True)

        # Attention mechanism
        self.attention = Attention(hidden_size)

        # Output layer with dropout
        self.out_dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_char, hidden, encoder_outputs):
        # input_char: [batch_size] (current input token indices)
        # hidden: (h_n, c_n) for LSTM or h_n for GRU/RNN
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        batch_size = input_char.size(0)

        # Embed input character and apply dropout
        embedded = self.embedding(input_char.squeeze(1)).unsqueeze(
            1)  # [batch_size, 1, embedding_dim]
        embedded = self.dropout(embedded)

        # Get the last hidden state from hidden (handle LSTM tuple)
        if self.cell_type == 'LSTM':
            last_hidden = hidden[0][-1]  # [batch_size, hidden_size]
        else:
            last_hidden = hidden[-1]     # [batch_size, hidden_size]

        # Calculate attention weights and context vector
        attn_weights = self.attention(
            last_hidden, encoder_outputs)  # [batch_size, seq_len]

        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, seq_len]

        # Compute context vector as weighted sum of encoder outputs
        # [batch_size, 1, hidden_size]
        context = torch.bmm(attn_weights, encoder_outputs)

        # Concatenate embedded input and context vector

        # [batch_size, 1, embedding_dim + hidden_size]
        rnn_input = torch.cat((embedded, context), dim=2)

        # Pass through RNN
        output, hidden = self.rnn(rnn_input, hidden)

        # Output shape: [batch_size, 1, hidden_size]
        output = self.out_dropout(output)
        output = self.out(output.squeeze(1))  # [batch_size, output_size]

        return F.log_softmax(output, dim=1), hidden, attn_weights.squeeze(1)


def beam_search_decode(model, src, sos_idx, eos_idx, max_len=30, beam_width=3, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Encode input
        encoder_outputs, encoder_hidden = model.encoder(src)

        # Prepare initial decoder hidden state
        if model.bidirectional:
            if model.cell_type == 'LSTM':
                h_n, c_n = encoder_hidden
                h_dec = torch.zeros(model.decoder.num_layers,
                                    1, model.decoder.hidden_size, device=device)
                c_dec = torch.zeros(model.decoder.num_layers,
                                    1, model.decoder.hidden_size, device=device)
                for layer in range(model.encoder.num_layers):
                    h_combined = torch.cat(
                        (h_n[2*layer], h_n[2*layer+1]), dim=1)
                    c_combined = torch.cat(
                        (c_n[2*layer], c_n[2*layer+1]), dim=1)
                    h_dec[layer] = model.hidden_transform(h_combined)
                    c_dec[layer] = model.hidden_transform(c_combined)
                decoder_hidden = (h_dec, c_dec)
            else:
                decoder_hidden = torch.zeros(
                    model.decoder.num_layers, 1, model.decoder.hidden_size, device=device)
                for layer in range(model.encoder.num_layers):
                    h_combined = torch.cat(
                        (encoder_hidden[2*layer], encoder_hidden[2*layer+1]), dim=1)
                    decoder_hidden[layer] = model.hidden_transform(h_combined)
        else:
            decoder_hidden = encoder_hidden

        # Beam search initialization
        # (sequence, cumulative log-prob, hidden)
        beams = [([sos_idx], 0.0, decoder_hidden)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for seq, score, hidden in beams:
                if seq[-1] == eos_idx:
                    completed.append((seq, score))
                    continue
                input_char = torch.tensor([seq[-1]], device=device)

                # Decoder forward with attention requires encoder_outputs
                output, hidden_new, _attn = model.decoder(
                    input_char, hidden, encoder_outputs)

                # Already log_softmax, shape [batch_size=1, output_size]
                log_probs = output
                topk_log_probs, topk_indices = torch.topk(
                    log_probs.squeeze(0), beam_width)
                for k in range(beam_width):
                    next_seq = seq + [topk_indices[k].item()]
                    next_score = score + topk_log_probs[k].item()

                    # Make sure to detach hidden state to avoid graph buildup
                    if model.cell_type == 'LSTM':
                        h_new = tuple(h.detach() for h in hidden_new)
                        new_beams.append((next_seq, next_score, h_new))
                    else:
                        new_beams.append(
                            (next_seq, next_score, hidden_new.detach()))

            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[
                :beam_width]
            if not beams:
                break

        # Add any remaining beams ending with <eos>
        completed += [(seq, score)
                      for seq, score, _ in beams if seq[-1] == eos_idx]

        # If none ended with <eos>, just take the best
        if not completed:
            completed = beams

        # Sort by score
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        return completed

# Seq2Seq implements an Encoder and Decoder for end-to-end sequence-to-sequence modeling
# input_size: size of source vocabulary
# output_size: size of target vocabulary
# embedding_dim: dimension of embeddings in both encoder and decoder
# hidden_size: size of hidden states in encoder and decoder (must match for vanilla seq2seq)
# encoder_layers / decoder_layers: number of stacked RNN layers
# cell_type: 'GRU', 'LSTM', or 'RNN'
# dropout_p: dropout probability for embeddings and RNN layers
# bidirectional_encoder: if True, runs encoder bidirectionally and transforms hidden state


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim=256, hidden_size=256,
                 encoder_layers=1, decoder_layers=1, cell_type='GRU', dropout_p=0.2,
                 bidirectional_encoder=False):
        super(Seq2Seq, self).__init__()

        self.encoder = EncoderRNN(input_size, hidden_size, embedding_dim,
                                  num_layers=encoder_layers, cell_type=cell_type,
                                  dropout_p=dropout_p, bidirectional=bidirectional_encoder)

        self.bidirectional = bidirectional_encoder
        directions = 2 if bidirectional_encoder else 1

        if bidirectional_encoder:
            self.hidden_transform = nn.Linear(
                hidden_size * directions, hidden_size)

        self.decoder = DecoderRNNWithAttention(output_size, hidden_size, embedding_dim,
                                               num_layers=decoder_layers, cell_type=cell_type,
                                               dropout_p=dropout_p)

        self.cell_type = cell_type

    def _match_decoder_layers(self, hidden, batch_size):
        if hidden.size(0) > self.decoder.num_layers:
            return hidden[:self.decoder.num_layers]
        elif hidden.size(0) < self.decoder.num_layers:
            pad = torch.zeros(self.decoder.num_layers - hidden.size(0),
                              batch_size, self.decoder.hidden_size,
                              device=hidden.device)
            return torch.cat([hidden, pad], dim=0)
        else:
            return hidden

    def forward(self, src, trg, teacher_forcing_ratio=0.5, return_attention=False):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, trg_len, output_size).to(src.device)
        all_attentions = [] if return_attention else None

        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_hidden = None

        if self.bidirectional:
            if self.cell_type == 'LSTM':
                h_n, c_n = encoder_hidden
                h_dec = torch.zeros(
                    self.decoder.num_layers, batch_size, self.decoder.hidden_size).to(src.device)
                c_dec = torch.zeros(
                    self.decoder.num_layers, batch_size, self.decoder.hidden_size).to(src.device)

                for layer in range(self.decoder.num_layers):
                    enc_layer = min(layer, self.encoder.num_layers - 1)
                    h_combined = torch.cat(
                        (h_n[2 * enc_layer], h_n[2 * enc_layer + 1]), dim=1)
                    c_combined = torch.cat(
                        (c_n[2 * enc_layer], c_n[2 * enc_layer + 1]), dim=1)
                    h_dec[layer] = self.hidden_transform(h_combined)
                    c_dec[layer] = self.hidden_transform(c_combined)

                decoder_hidden = (h_dec, c_dec)

            else:
                h_n = encoder_hidden
                h_dec = torch.zeros(
                    self.decoder.num_layers, batch_size, self.decoder.hidden_size).to(src.device)
                for layer in range(self.decoder.num_layers):
                    enc_layer = min(layer, self.encoder.num_layers - 1)
                    h_combined = torch.cat(
                        (h_n[2 * enc_layer], h_n[2 * enc_layer + 1]), dim=1)
                    h_dec[layer] = self.hidden_transform(h_combined)
                decoder_hidden = h_dec

        else:
            if self.cell_type == "LSTM":
                h, c = encoder_hidden
                decoder_hidden = (
                    self._match_decoder_layers(h, batch_size),
                    self._match_decoder_layers(c, batch_size)
                )
            else:
                decoder_hidden = self._match_decoder_layers(
                    encoder_hidden, batch_size)

        input_char = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, decoder_hidden, attn_weights = self.decoder(
                input_char, decoder_hidden, encoder_outputs)
            outputs[:, t, :] = output

            if return_attention:
                # [batch, 1, src_len]
                all_attentions.append(attn_weights.unsqueeze(1))

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_char = trg[:, t].unsqueeze(1) if teacher_force else top1

        if return_attention:
            # Shape: [batch_size, trg_len-1, src_len]
            all_attentions = torch.cat(all_attentions, dim=1)
            return outputs, all_attentions

        return outputs


class LexiconDataset(Dataset):
    def __init__(self, path, src_vocab=None, tgt_vocab=None, build_vocab=False):
        self.pairs = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) < 2:
                    continue
                # Telugu is first column, romanized second
                tgt, src = cols[0], cols[1]
                self.pairs.append((src, tgt))  # src = romanized, tgt = telugu

        if build_vocab:
            self.src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
            self.tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
            for rom, dev in self.pairs:
                for c in rom:
                    self.src_vocab.setdefault(c, len(self.src_vocab))
                for c in dev:
                    self.tgt_vocab.setdefault(c, len(self.tgt_vocab))
        else:
            assert src_vocab and tgt_vocab, "Must provide vocabs if not building."
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rom, dev = self.pairs[idx]
        src_idxs = [self.src_vocab.get(
            c, self.src_vocab['<unk>']) for c in rom]
        tgt_idxs = [self.tgt_vocab['<sos>']] + [self.tgt_vocab.get(
            c, self.tgt_vocab['<unk>']) for c in dev] + [self.tgt_vocab['<eos>']]
        return torch.tensor(src_idxs, dtype=torch.long), torch.tensor(tgt_idxs, dtype=torch.long)


def collate_fn(batch):
    """
    Pads all src/tgt sequences in the batch to the max length.
    Returns:
      padded_src: (batch_size, max_src_len)
      padded_tgt: (batch_size, max_tgt_len)
    """
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)

    padded_src = torch.full((len(batch), max_src), 0, dtype=torch.long)
    padded_tgt = torch.full((len(batch), max_tgt), 0, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        padded_src[i, :len(s)] = s
        padded_tgt[i, :len(t)] = t

    return padded_src, padded_tgt


def get_dataloaders(base_dir, batch_size, build_vocab=False):
    """
    Returns:
      train_loader, val_loader, test_loader,
      src_vocab_size, tgt_vocab_size, pad_index, src_vocab, tgt_vocab
    """
    train_p = os.path.join(base_dir, 'te.translit.sampled.train.tsv')
    dev_p = os.path.join(base_dir, 'te.translit.sampled.dev.tsv')
    test_p = os.path.join(base_dir, 'te.translit.sampled.test.tsv')

    train_ds = LexiconDataset(train_p, build_vocab=build_vocab)
    src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab
    val_ds = LexiconDataset(dev_p,  src_vocab, tgt_vocab)
    test_ds = LexiconDataset(test_p, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader = DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds,  batch_size=1,
                             shuffle=False, collate_fn=collate_fn)

    return (train_loader, val_loader, test_loader,
            len(src_vocab), len(tgt_vocab), src_vocab['<pad>'],
            src_vocab, tgt_vocab)


class EarlyStopper:
    """Stops a run if the monitored metric doesn’t improve for `patience` steps."""

    def __init__(self, patience=5, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best = 0, None

    def should_stop(self, current):
        if self.best is None or current > self.best + self.min_delta:
            self.best, self.counter = current, 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, loader, opt, crit, device, teacher):
    model.train()
    total = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        out = model(src, tgt, teacher_ratio=teacher)
        loss = crit(out.view(-1, out.size(-1)), tgt.view(-1))
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)


def eval_epoch(model, loader, pad_idx, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt, teacher_ratio=0.0).argmax(2)
            for p, t in zip(out, tgt):
                mask = t[1:] != pad_idx
                if torch.equal(p[1:][mask], t[1:][mask]):
                    correct += 1
                total += 1
    return correct/total


def infer_and_save(model, loader, pad_idx, tgt_vocab, out_dir):
    idx2char = {i: c for c, i in tgt_vocab.items()}
    records = []
    model.eval()
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            out = model(src, tgt, teacher_ratio=0.0).argmax(2)[
                0].cpu().tolist()
            true = tgt[0].cpu().tolist()
            pred_s = ''.join(idx2char[i] for i in out[1:] if i != pad_idx)
            true_s = ''.join(idx2char[i] for i in true[1:] if i != pad_idx)
            records.append(
                {'Input': None, 'True': true_s, 'Predicted': pred_s})
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(f"{out_dir}/predictions.csv", index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} samples to {out_dir}/predictions.csv")

# --- CLI -------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_dir', required=True)
    p.add_argument(
        '--mode', choices=['train', 'eval', 'predict'], required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--emb_dim',   type=int, default=64)
    p.add_argument('--hidden',    type=int, default=128)
    p.add_argument('--enc_layers', type=int, default=3)
    p.add_argument('--dec_layers', type=int, default=3)
    p.add_argument(
        '--cell',      choices=['GRU', 'LSTM', 'RNN'], default='GRU')
    p.add_argument('--dropout',   type=float, default=0.3)
    p.add_argument('--bidir',     action='store_true')
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--epochs',    type=int, default=10)
    p.add_argument('--teacher',   type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl, val_dl, test_dl, src_vocab, tgt_vocab = get_dataloaders(
        args.base_dir, args.batch_size, build_vocab=(args.mode == 'train')
    )
    pad_idx = src_vocab['<pad>']
    model = Seq2Seq(args, len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss(ignore_index=pad_idx)

    if args.mode == 'train':
        best = 0
        for ep in range(1, args.epochs+1):
            loss = train_epoch(model, train_dl, optimizer,
                               criterion, device, args.teacher)
            acc = eval_epoch(model, val_dl, pad_idx, device)
            print(
                f"[Epoch {ep}/{args.epochs}] loss={loss:.4f} val_acc={acc:.4f}")
            if acc > best:
                best = acc
                torch.save(model.state_dict(), 'best_model.pt')
    elif args.mode == 'eval':
        model.load_state_dict(torch.load('best_model.pt'))
        acc = eval_epoch(model, test_dl, pad_idx, device)
        print(f"Test accuracy = {acc:.4f}")
    else:  # predict
        model.load_state_dict(torch.load('best_model.pt'))
        infer_and_save(model, test_dl, pad_idx, tgt_vocab,
                       out_dir='predictions_attention')


if __name__ == '__main__':
    main()
