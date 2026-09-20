from pathlib import Path
import tempfile
import unittest

import torch

from app import batch, loss_on, read_data
from model import Config, TextModel, load_checkpoint, save_checkpoint


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = TextModel(Config(context=16, width=32, layers=1, heads=2, dropout=0))

    def test_future_tokens_cannot_change_past_predictions(self):
        self.model.eval()
        original = torch.randint(256, (1, 16))
        changed = original.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 256
        with torch.no_grad():
            torch.testing.assert_close(self.model(original)[:, :8], self.model(changed)[:, :8])

    def test_model_learns_and_checkpoint_restores_predictions(self):
        data = torch.tensor(list(b'abcd' * 200), dtype=torch.uint8)
        x, y = batch(data, 8, 16, 'cpu')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        initial = loss_on(self.model, x, y).item()
        for _ in range(60):
            optimizer.zero_grad()
            loss = loss_on(self.model, x, y)
            loss.backward()
            optimizer.step()
        self.assertLess(loss_on(self.model, x, y).item(), initial * 0.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(path, self.model, optimizer, 60, 'hash')
            restored, checkpoint = load_checkpoint(path, 'cpu')
            self.assertEqual(checkpoint['step'], 60)
            torch.testing.assert_close(self.model(x), restored(x))
            resumed_optimizer = torch.optim.AdamW(restored.parameters())
            resumed_optimizer.load_state_dict(checkpoint['optimizer'])
            self.assertTrue(resumed_optimizer.state)

    def test_batches_are_shifted_and_stay_in_split(self):
        data = torch.arange(100, dtype=torch.uint8)
        x, y = batch(data, 20, 16, 'cpu')
        torch.testing.assert_close(x + 1, y)
        self.assertLess(y.max().item(), 100)

    def test_data_split_and_small_corpus_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            path.write_text('abcd' * 100, encoding='utf-8')
            training, validation, fingerprint = read_data(path, 16)
            self.assertEqual((len(training), len(validation)), (360, 40))
            self.assertEqual(len(fingerprint), 64)
            with self.assertRaises(ValueError):
                read_data(path, 100)

    def test_generation_handles_long_unicode_prompt(self):
        prompt = 'Hello \u0562' * 10
        result = self.model.generate(prompt, count=5, top_k=1)
        self.assertTrue(result.startswith(prompt))
        self.assertGreater(len(result), len(prompt))
        with self.assertRaises(ValueError):
            self.model.generate('')


if __name__ == '__main__':
    unittest.main()
