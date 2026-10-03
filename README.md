<div align="center">

# localML

### A small language model. Built from scratch. Trained on your text.

**Python · PyTorch · Byte-level Transformer · Local inference**

Explore how a language model learns, one byte at a time.

[Quick start](#quick-start) · [Your own data](#train-on-your-own-text) · [How it works](#how-it-works) · [Troubleshooting](#troubleshooting)

</div>

---

## Meet the model

localML is a compact causal Transformer with its architecture, training loop, and generation code in this project. It starts with random weights and learns patterns from UTF-8 text. PyTorch provides the tensor operations, automatic differentiation, and GPU kernels.

| From scratch | Under your control | Offline after setup |
| :--- | :--- | :--- |
| Random initial weights; no pretrained model downloads | Choose the text, settings, and checkpoints | Train and generate locally without a model API |

> [!IMPORTANT]
> **This is a text continuation experiment, not a trained coding assistant.** The demo corpus contains short Python examples. Generated text may resemble code without being correct or runnable; the model does not understand requests or follow instructions.

## Quick start

Run these commands in **PowerShell**, from the project directory. The documented Python version range is **3.10–3.14**.

### 1. Set up the environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Already have the environment installed? Skip this step.

The requirements pin PyTorch `2.11.0+cu128` from the CUDA 12.8 package index. The initial installation is a large download; it installs computing libraries, not trained model weights. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform-specific setup.

### 2. Train a model

```powershell
.\.venv\Scripts\python.exe app.py train --steps 2000
```

This trains on `data/demo.txt` and saves to `checkpoints/text.pt`. Progress is printed every 100 steps. CUDA is selected when available; otherwise, training uses the CPU.

### 3. Generate text

```powershell
.\.venv\Scripts\python.exe app.py generate --prompt "def " --length 300 --temperature 0.5
```

Generation uses `checkpoints/text.pt` by default. Replace the prompt to explore other continuations.

The training corpus is a small collection of Python examples, so use this as an experiment with code-shaped text. A larger, high-quality code corpus is needed for more useful results. Training on a new corpus does not change an existing checkpoint; choose a new checkpoint path or start fresh when changing the data.

## Train on your own text

Place UTF-8 `.txt` files in `data/my_text/`. The loader includes text files in subdirectories and combines them in sorted path order.

```text
data/
├── demo.txt
└── my_text/
    ├── notes.txt
    └── stories.txt
```

Start a new checkpoint for your dataset:

```powershell
.\.venv\Scripts\python.exe app.py train --data data/my_text --steps 10000 --checkpoint checkpoints/my_text.pt
```

Then generate from it:

```powershell
.\.venv\Scripts\python.exe app.py generate --checkpoint checkpoints/my_text.pt --prompt "Once upon a time" --length 500 --temperature 0.5
```

Clean, varied text gives the model more to learn than the tiny demo corpus. A few megabytes is a better starting experiment, but more data and steps do not guarantee useful conversation. All training data is loaded into system RAM.

### Pause and resume

Press **Ctrl+C during training** to save the current weights. Continue with:

```powershell
.\.venv\Scripts\python.exe app.py train --data data/my_text --checkpoint checkpoints/my_text.pt --resume --steps 2000
```

- `--steps` is the number of **additional** training steps.
- Resume requires the same training data and restores model and optimizer state.
- The checkpoint determines the architecture; architecture flags are ignored on resume.
- Saves happen at evaluation intervals, at completion, and when training handles Ctrl+C. The latest checkpoint is retained, not the best validation checkpoint.
- Sampling restarts from `--seed`; resume does not restore the exact random stream.

> [!TIP]
> An existing checkpoint is protected against accidental fresh training. Use `--resume` to continue it, or choose a new checkpoint path to start over.

## How it works

```text
UTF-8 text → bytes → causal Transformer → next-byte prediction
                              ↑                    │
                              └── training loss ───┘

Prompt → sample one byte → append → repeat → decode text
```

Each token is a byte with a value from 0 to 255. There is no external tokenizer or learned vocabulary. The model predicts the next byte using only preceding bytes.

| Default architecture | Value |
| :--- | :--- |
| Parameters | 3,356,160 |
| Transformer blocks | 4 |
| Embedding width | 256 |
| Attention heads | 4 |
| Context window | 256 bytes |
| Training batch size | 16 |
| Vocabulary | 256 byte values |

**Bytes are not words.** Some Unicode characters take several bytes. During generation, only the most recent context window is used. An undertrained model can emit invalid UTF-8 sequences, displayed as replacement characters.

### Reading the training output

```text
step 200 | train loss 1.8375 | validation loss 2.4221 | 2.7s
```

Training loss measures prediction error on a training batch. Validation loss is evaluated on windows sampled from the final 10% of the combined corpus. Lower validation loss indicates better next-byte prediction on those held-out examples; it is not a measure of conversational ability.

If training loss falls while validation loss rises, the model is fitting its training text without improving on unseen text. Repeated material across the split can also make validation results look overly optimistic.

## Useful controls

| Option | Applies to | Purpose |
| :--- | :--- | :--- |
| `--checkpoint PATH` | Both | Select the weights to save or load |
| `--device auto\|cpu\|cuda` | Both | Choose the compute device; default: `auto` |
| `--seed 42` | Both | Set the random seed |
| `--steps 2000` | Training | Set additional optimizer steps |
| `--batch-size 4` | Training | Reduce memory usage compared with the default 16 |
| `--eval-every 100` | Training | Set the evaluation and checkpoint interval |
| `--resume` | Training | Continue an existing checkpoint |
| `--length 300` | Generation | Set the number of new bytes to sample |
| `--temperature 0.5` | Generation | Sample more conservatively than the default 0.8 |
| `--top-k 40` | Generation | Sample among the highest-scoring byte candidates |

See all options:

```powershell
.\.venv\Scripts\python.exe app.py train --help
.\.venv\Scripts\python.exe app.py generate --help
```

## Troubleshooting

| What you see | What to do |
| :--- | :--- |
| Jumbled letters or broken words | Check that you loaded the intended checkpoint and trained it sufficiently. Try a lower temperature; it cannot compensate for missing training. |
| Output is not valid code | The demo corpus is small and the model predicts text one byte at a time. Add more high-quality code examples and train a new checkpoint; generated code still needs review. |
| Missing checkpoint | Train first, or supply the path to an existing checkpoint with `--checkpoint`. |
| “Checkpoint exists” | Add `--resume`, or select a new output path. |
| “Training data changed” | Resume with the original data, or train a new checkpoint on the changed corpus. |
| CUDA out of memory | Reduce `--batch-size` to 4, then lower it further if needed. |
| CUDA unavailable | Check the PyTorch installation, or use `--device cpu`. |
| Corpus too small | Add text or reduce `--context` for a new model; both splits must exceed the context length. |

## Project map

```text
localML/
├── app.py               # Data loading, training, validation, and CLI
├── model.py             # Transformer, sampling, and checkpoint handling
├── requirements.txt     # PyTorch and NumPy dependencies
├── data/
│   └── demo.txt         # Short Python examples
├── tests/
│   └── test_model.py    # Learning, causality, data, and generation checks
└── checkpoints/         # Local weights; ignored by Git
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The five tests cover causal attention, learning a small pattern, checkpoint restoration, data boundaries, and generation with a Unicode prompt.

Local training and generation have been verified with **PyTorch 2.11.0+cu128** on an **RTX 5060 Laptop GPU**. The extra 2,000-step demo training run took approximately **57 seconds** on this workspace's machine. This is an observed demo run, not a performance guarantee for other hardware, datasets, or settings.
