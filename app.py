"""Train your own text generator from scratch, then generate text locally."""
import argparse
import hashlib
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from model import Config, TextModel, load_checkpoint, save_checkpoint


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable. Install the CUDA PyTorch build or use --device cpu.')
    return torch.device(name)


def read_data(path, context):
    path = Path(path)
    files = sorted(path.rglob('*.txt')) if path.is_dir() else [path]
    if not files:
        raise ValueError('No .txt files found.')
    raw = '\n\n'.join(file.read_text(encoding='utf-8') for file in files).encode('utf-8')
    split = int(len(raw) * 0.9)
    if min(split, len(raw) - split) <= context:
        raise ValueError(f'Need at least about {(context + 1) * 10 + 10:,} UTF-8 bytes for the train/validation split.')
    data = torch.tensor(list(raw), dtype=torch.uint8)
    return data[:split], data[split:], hashlib.sha256(raw).hexdigest()


def batch(data, batch_size, context, device, generator=None):
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator)
    indices = starts[:, None] + torch.arange(context + 1)[None, :]
    chunk = data[indices].to(device=device, dtype=torch.long)
    return chunk[:, :-1], chunk[:, 1:]


def loss_on(model, x, y):
    return F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    # Repeatable held-out windows; evaluation does not consume training RNG state.
    generator = torch.Generator().manual_seed(123)
    losses = [loss_on(model, *batch(data, batch_size, model.config.context, device, generator)).item()
              for _ in range(5)]
    model.train()
    return sum(losses) / len(losses)


def train(args, device):
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.lr <= 0:
        raise ValueError('Steps, batch size, evaluation interval and learning rate must be positive.')
    destination = Path(args.checkpoint)
    if args.resume:
        model, saved = load_checkpoint(destination, device)
    else:
        if destination.exists():
            raise ValueError('Checkpoint exists. Use --resume or choose a new --checkpoint path.')
        model = TextModel(Config(context=args.context, width=args.width, layers=args.layers, heads=args.heads)).to(device)
        saved = {'step': 0}
    training, validation, data_hash = read_data(args.data, model.config.context)
    if args.resume and saved['data_hash'] != data_hash:
        raise ValueError('Training data changed. Resume requires the same data and split.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters | {device} | '
          f'{len(training):,} training bytes, {len(validation):,} validation bytes', flush=True)
    if len(training) < 100_000:
        print('Small demo corpus: expect memorization and rough output. Add more text for useful results.', flush=True)
    model.train()
    step = saved['step']
    started = time.monotonic()
    try:
        for _ in range(args.steps):
            x, y = batch(training, args.batch_size, model.config.context, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(model, x, y)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss. Try a lower learning rate.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.eval_every == 0 or step == saved['step'] + args.steps:
                val = evaluate(model, validation, args.batch_size, device)
                print(f'step {step} | train loss {loss.item():.4f} | validation loss {val:.4f} | '
                      f'{time.monotonic() - started:.1f}s', flush=True)
                save_checkpoint(destination, model, optimizer, step, data_hash)
    except KeyboardInterrupt:
        print('\nInterrupted; saving current weights.', flush=True)
        save_checkpoint(destination, model, optimizer, step, data_hash)
    print(f'Saved {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train', help='Learn next-byte prediction from UTF-8 text.')
    training.add_argument('--data', default='data/demo.txt')
    training.add_argument('--steps', type=int, default=2000, help='Additional optimizer steps.')
    training.add_argument('--batch-size', type=int, default=16)
    training.add_argument('--context', type=int, default=256)
    training.add_argument('--width', type=int, default=256)
    training.add_argument('--layers', type=int, default=4)
    training.add_argument('--heads', type=int, default=4)
    training.add_argument('--lr', type=float, default=3e-4)
    training.add_argument('--eval-every', type=int, default=100)
    training.add_argument('--resume', action='store_true')
    generation = sub.add_parser('generate', help='Continue a prompt using your trained weights.')
    generation.add_argument('--prompt', default='The ')
    generation.add_argument('--length', type=int, default=300, help='Number of new UTF-8 bytes.')
    generation.add_argument('--temperature', type=float, default=0.8)
    generation.add_argument('--top-k', type=int, default=40)
    for command in (training, generation):
        command.add_argument('--checkpoint', default='checkpoints/text.pt')
        command.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        command.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    try:
        device = choose_device(args.device)
        if args.command == 'train':
            train(args, device)
        else:
            model, _ = load_checkpoint(args.checkpoint, device)
            print(model.generate(args.prompt, args.length, args.temperature, args.top_k))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
