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
import argparse
import json

# DEVICE
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

# DecoderRNN generates target sequences one token at a time


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_dim, num_layers=1,
                 cell_type='GRU', dropout_p=0.1):

        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # Embedding layer for target characters
        self.embedding = nn.Embedding(output_size, embedding_dim)

        # Dropout applied before RNN
        self.dropout = nn.Dropout(dropout_p)

        dropout_p = dropout_p if num_layers > 1 else 0

        # RNN layer
        if cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_size,
                              num_layers, dropout=dropout_p, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_size,
                               num_layers, dropout=dropout_p, batch_first=True)
        else:  # Default to RNN
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers,
                              dropout=dropout_p, nonlinearity='tanh', batch_first=True)

        # Output projection layer with dropout
        self.out_dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    # Forward pass for a single decoding step
    def forward(self, input_char, hidden):
        # Convert input to embeddings and apply dropout
        embedded = self.embedding(input_char)  # [batch_size, 1, embedding_dim]
        embedded = self.dropout(embedded)

        # Pass through RNN
        output, hidden = self.rnn(embedded, hidden)

        # Apply dropout before prediction layer
        output = self.out_dropout(output)
        output = self.out(output[:, 0, :])

        return F.log_softmax(output, dim=1), hidden

#  Perform beam search decoding with the trained seq2seq model.


def beam_search_decode(model, src, sos_idx, eos_idx, max_len=30, beam_width=3, device='gpu'):
    model.eval()
    with torch.no_grad():
        # Encode input
        encoder_outputs, encoder_hidden = model.encoder(src)

        # Prepare initial decoder hidden state
        if model.bidirectional:
            if model.cell_type == 'LSTM':
                h_n, c_n = encoder_hidden
                h_dec = torch.zeros(model.decoder.num_layers,
                                    1, model.decoder.hidden_size).to(device)
                c_dec = torch.zeros(model.decoder.num_layers,
                                    1, model.decoder.hidden_size).to(device)
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
                    model.decoder.num_layers, 1, model.decoder.hidden_size).to(device)
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
                input_char = torch.tensor([[seq[-1]]], device=device)
                output, hidden_new = model.decoder(input_char, hidden)
                log_probs = output.squeeze(0)  # [output_size]
                topk_log_probs, topk_indices = torch.topk(
                    log_probs, beam_width)
                for k in range(beam_width):
                    next_seq = seq + [topk_indices[k].item()]
                    next_score = score + topk_log_probs[k].item()
                    new_beams.append((next_seq, next_score, hidden_new))
            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[
                :beam_width]
            if not beams:
                break

        # Add any remaining beams
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

        # Create encoder RNN
        self.encoder = EncoderRNN(input_size, hidden_size, embedding_dim,
                                  num_layers=encoder_layers, cell_type=cell_type,
                                  dropout_p=dropout_p, bidirectional=bidirectional_encoder)

        self.bidirectional = bidirectional_encoder
        directions = 2 if bidirectional_encoder else 1

        # If bidirectional encoder, need a linear layer to transform hidden state
        if bidirectional_encoder:
            self.hidden_transform = nn.Linear(
                hidden_size * directions, hidden_size)

        # Create decoder RNN
        self.decoder = DecoderRNN(output_size, hidden_size, embedding_dim,
                                  num_layers=decoder_layers, cell_type=cell_type,
                                  dropout_p=dropout_p)

        self.cell_type = cell_type

    def _match_decoder_layers(self, hidden, batch_size):
        """Ensures hidden state matches decoder layers by trimming or padding."""
        if hidden.size(0) > self.decoder.num_layers:
            return hidden[:self.decoder.num_layers]
        elif hidden.size(0) < self.decoder.num_layers:
            pad = torch.zeros(self.decoder.num_layers - hidden.size(0),
                              batch_size, self.decoder.hidden_size,
                              device=hidden.device)
            return torch.cat([hidden, pad], dim=0)
        else:
            return hidden

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_size = self.decoder.output_size

        # Tensor to store decoder outputs (logits or log-probs)
        outputs = torch.zeros(batch_size, trg_len, output_size).to(src.device)

        # Encode the source sequence
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_hidden = None

        # Prepare initial hidden state for decoder
        if self.bidirectional:
            # Bidirectional encoder returns hidden states with doubled layers * 2 directions

            if self.cell_type == 'LSTM':
                # For LSTM, hidden state is a tuple (h_n, c_n)
                h_n, c_n = encoder_hidden

                # Initialize decoder hidden states
                h_dec = torch.zeros(
                    self.decoder.num_layers, batch_size, self.decoder.hidden_size).to(src.device)
                c_dec = torch.zeros(
                    self.decoder.num_layers, batch_size, self.decoder.hidden_size).to(src.device)

                # For each decoder layer, combine corresponding forward and backward encoder layers
                for layer in range(self.decoder.num_layers):
                    # Clamp to max encoder layers index to avoid index errors if decoder has more layers
                    enc_layer = min(layer, self.encoder.num_layers - 1)

                    # Concatenate forward and backward hidden states from encoder for this layer
                    h_combined = torch.cat(
                        (h_n[2 * enc_layer], h_n[2 * enc_layer + 1]), dim=1)
                    c_combined = torch.cat(
                        (c_n[2 * enc_layer], c_n[2 * enc_layer + 1]), dim=1)

                    # Transform concatenated states to decoder hidden size
                    h_dec[layer] = self.hidden_transform(h_combined)
                    c_dec[layer] = self.hidden_transform(c_combined)

                decoder_hidden = (h_dec, c_dec)

            else:
                # For GRU or vanilla RNN (hidden state is a single tensor)
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

        # First input to decoder is <sos> token from target
        input_char = trg[:, 0].unsqueeze(1)  # shape: (batch_size, 1)

        # Decode one token at a time
        for t in range(1, trg_len):
            output, decoder_hidden = self.decoder(input_char, decoder_hidden)
            outputs[:, t, :] = output

            # Decide if teacher forcing should be used
            teacher_force = random.random() < teacher_forcing_ratio

            # Get highest probability token from output
            top1 = output.argmax(1).unsqueeze(1)

            # Next input is either true target token or predicted token
            input_char = trg[:, t].unsqueeze(1) if teacher_force else top1

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


# --- Training / Evaluation -------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, teacher_ratio):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        out = model(src, tgt, teacher_ratio)
        loss = criterion(out.view(-1, out.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, pad_idx, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt, teacher_ratio=0.0)
            pred = out.argmax(-1)
            for p, t in zip(pred, tgt):
                mask = t[1:] != pad_idx
                if torch.equal(p[1:][mask], t[1:][mask]):
                    correct += 1
                total += 1
    return correct / total


def predict_all(model, loader, pad_idx, tgt_vocab, device):
    model.eval()
    idx2char = {i: c for c, i in tgt_vocab.items()}
    samples = []
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt, teacher_ratio=0.0)
            pred = out.argmax(-1)[0]
            true = tgt[0]
            pred_str = ''.join(idx2char[i.item()]
                               for i in pred[1:] if i.item() != pad_idx)
            true_str = ''.join(idx2char[i.item()]
                               for i in true[1:] if i.item() != pad_idx)
            samples.append((pred_str, true_str))
    return samples


def save_predictions(samples, out_dir, filename="predictions.csv"):

    os.makedirs("predictions_vanilla", exist_ok=True)

    df1 = pd.DataFrame(samples)
    df1.to_csv("predictions_vanilla/predictions.csv",
               index=False, encoding="utf-8-sig")

    print(" Saved all predictions to predictions_vanilla/predictions.csv")


# --- CLI Parser & Main -----------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Seq2Seq transliteration")
    p.add_argument("--base_dir", type=str, required=True,
                   help="path to dakshina dataset")
    p.add_argument(
        "--mode", choices=["train", "eval", "predict"], required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--enc_layers", type=int, default=3)
    p.add_argument("--dec_layers", type=int, default=3)
    p.add_argument("--cell", choices=["GRU", "LSTM", "RNN"], default="GRU")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--bidir", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--teacher_ratio", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(args.base_dir, args.batch_size,
                        build_vocab=(args.mode == "train"))

    model = Seq2Seq(args, len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss(ignore_index=src_vocab['<pad>'])

    if args.mode == "train":
        best_acc = 0
        for ep in range(1, args.epochs+1):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, args.teacher_ratio)
            val_acc = eval_epoch(model, val_loader, src_vocab['<pad>'], device)
            print(
                f"Epoch {ep}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_seq2seq.pth")

    elif args.mode == "eval":
        model.load_state_dict(torch.load("best_seq2seq.pth"))
        acc = eval_epoch(model, test_loader, src_vocab['<pad>'], device)
        print(f"Test accuracy = {acc:.4f}")

    else:  # predict
        model.load_state_dict(torch.load("best_seq2seq.pth"))
        samples = predict_all(model, test_loader,
                              src_vocab['<pad>'], tgt_vocab, device)
        df = pd.DataFrame(samples, columns=["Predicted", "True"])
        print(df.sample(10).to_markdown(index=False))
        save_predictions(samples, out_dir="predictions_vanilla")


if __name__ == "__main__":
    main()
