# da6401_assignment3

# Project Overview

This project demonstrates how to build and run a character‑level Seq2Seq model with additive (Bahdanau) attention in PyTorch. We use the Dakshina transliteration dataset (Romanized ↔ Telugu lexicons) to train an encoder‑decoder architecture:

We train a model with attention and other without attention

Encoder: Embedding → GRU/LSTM/RNN → hidden outputs
Attention: Computes alignment weights between decoder hidden state and all encoder outputs
Decoder: Embedding + context vector → GRU/LSTM/RNN → log‑softmax predictions

# Usage
All functionality is exposed via the CLI in train_with_attention.py and train_without_attention.py

- Command‑Line Interface
```
usage: train_with_attention.py [-h] --base_dir BASE_DIR --mode {train,eval,predict}
                        [--batch_size BATCH_SIZE] [--emb_dim EMB_DIM]
                        [--hidden HIDDEN]
                        [--enc_layers ENCODER_LAYERS] [--dec_layers DECODER_LAYERS]
                        [--cell {GRU,LSTM,RNN}] [--dropout DROPOUT] [--bidir]
                        [--lr LR] [--epochs EPOCHS] [--teacher TEACHER]
```

# Modes
- Training
```
python train_seq2seq.py \
  --mode train \
  --base_dir data/te/lexicons \
  --batch_size 64 \
  --emb_dim 64 \
  --hidden 128 \
  --enc_layers 3 \
  --dec_layers 3 \
  --cell GRU \
  --dropout 0.3 \
  --bidir \
  --lr 1e-3 \
  --epochs 10 \
  --teacher 0.7
```
Builds vocabulary on *.train.tsv.
Trains with teacher‑forcing ratio = 0.7.

- Evaluation
```
python train_seq2seq.py \
  --mode eval \
  --base_dir data/te/lexicons
```

- Computes sequence‑level accuracy on the test set.

Prediction
```
python train_seq2seq.py \
  --mode predict \
  --base_dir data/te/lexicons
```

Runs forward pass with no teacher forcing.
Saves all test outputs to predictions_attention/predictions.csv with columns:
True: ground-truth string
Predicted: model’s transliteration

## Model Components

- **EncoderRNN** (`model.encoder`)  
  - Character embedding  
  - RNN cell (GRU / LSTM / vanilla RNN)  
  - Optional bidirectional layers  

- **Attention** (`model.decoder.attn`)  
  - Computes additive scores:  
    \`\`\`  
    score = vᵀ · tanh(W · [h_t; h_s])  
    \`\`\`  
  - Applies softmax over encoder time steps  

- **DecoderWithAttention** (`model.decoder`)  
  - Embedding of the previous output token  
  - Attention context vector  
  - RNN cell + linear projection + log‑softmax  

- **Beam Search** (inference utility)  
  - Keeps top‑k hypotheses per decoding step  

---

## Configuration

All hyperparameters are exposed as CLI flags:

| Flag            | Default | Description                                        |
|-----------------|---------|----------------------------------------------------|
| `--batch_size`  | 64      | Mini‑batch size                                    |
| `--emb_dim`     | 64      | Embedding dimension                                |
| `--hidden`      | 128     | Hidden state dimension                             |
| `--enc_layers`  | 3       | Number of encoder layers                           |
| `--dec_layers`  | 3       | Number of decoder layers                           |
| `--cell`        | `GRU`   | RNN cell type (`GRU` / `LSTM` / `RNN`)             |
| `--dropout`     | 0.3     | Dropout probability                                |
| `--bidir`       | –       | Use bidirectional encoder                          |
| `--lr`          | 1e-3    | Learning rate (Adam)                               |
| `--epochs`      | 10      | Maximum training epochs                            |
| `--teacher`     | 0.7     | Teacher‑forcing ratio                              |

---

## Outputs


- **predictions_attention/predictions_attention.csv and predictions_vanilla/predictions.csv**  
  CSV file containing **all** test predictions with columns:
  - `True`  
  - `Predicted`  

---

## Customization

- **Language**  
  Point `--base_dir` to another Dakshina dataset folder (e.g., Hindi).

- **Non‑Attention Variant**  
  Swap `DecoderWithAttention` for a basic `DecoderRNN` to compare results.

- **Beam Search**  
  Invoke `beam_search_decode()` for higher‑quality (but slower) inference.


# Directory Structure
```
├── README.md
├── train_with_attention.py      ]
├── train_without_attention.py
├── dl-assignment-3-without-attention.ipynb
├── dl-assignment3-withattention.ipynb   
├── predictions_attention/  # Default output folder for predictions
│   └── predictions_attention.csv
├── predictions_vanilla/  # Default output folder for predictions
│   └── predictions.csv
```
