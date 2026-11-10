"""Focused tests for the bean transfer-learning workflow."""

import tempfile
import unittest
import argparse
import json
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn

import beans_transfer as bt


class DatasetTests(unittest.TestCase):
    def test_dataset_splits_and_image(self):
        data = bt.load_metadata(bt.DEFAULT_DATA)
        self.assertEqual(data[4], [[345, 348, 341], [44, 45, 44], [43, 43, 42]])
        training = bt.BeansDataset(data, 0, bt.preprocessing("resnet18"))
        with mock.patch.object(Image, "open", wraps=Image.open) as open_image:
            tensor, label = training[0]
        self.assertEqual(open_image.call_count, 1)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(label, int(data[2][training.indices[0]]))
        self.assertTrue(np.all(data[3][training.indices] == 0))

    def test_invalid_offsets_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, jpeg=np.array([1, 2], dtype=np.uint8),
                     offsets=np.array([0, 0]), labels=np.array([0]), split=np.array([0]))
            with self.assertRaisesRegex(ValueError, "offsets"):
                bt.load_metadata(path)


class ModelTests(unittest.TestCase):
    def test_models_output_shapes_and_trainable_modes(self):
        torch.set_num_threads(2)
        for name in bt.MODEL_NAMES:
            with self.subTest(model=name):
                model, head, block = bt.create_model(name, pretrained=False)
                bt.set_trainable(model, head, block, False)
                head_ids = {id(p) for p in head.parameters()}
                self.assertEqual({id(p) for p in model.parameters() if p.requires_grad}, head_ids)
                bt.training_mode(model, head, block, False)
                self.assertFalse(model.training)
                with torch.no_grad():
                    self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape), (1, 3))
                bt.set_trainable(model, head, block, True)
                backbone_trainable = [p for p in block.parameters() if p.requires_grad]
                self.assertTrue(backbone_trainable)
                bt.training_mode(model, head, block, True)
                batch_norm = next(m for m in block.modules()
                                  if isinstance(m, nn.modules.batchnorm._BatchNorm))
                running_mean = batch_norm.running_mean.clone()
                before = next(p for p in block.parameters() if p.requires_grad).detach().clone()
                optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
                loss = nn.functional.cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 1]))
                # Randomly initialized CNNs can have tiny gradients in evaluation mode.
                (loss * 1_000_000).backward()
                optimizer.step()
                after = next(p for p in block.parameters() if p.requires_grad)
                self.assertFalse(torch.equal(before, after))
                torch.testing.assert_close(batch_norm.running_mean, running_mean)
                for module in model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        self.assertFalse(module.training)
                        self.assertTrue(all(not p.requires_grad for p in module.parameters()))
                allowed = {id(p) for p in block.parameters()} | head_ids
                self.assertTrue(all(id(p) in allowed for p in model.parameters() if p.requires_grad))

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        model, _, _ = bt.create_model("resnet18", pretrained=False)
        model.eval()
        sample = torch.rand(1, 3, 64, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(bt.checkpoint_payload(model, "resnet18", "frozen", {"seed": 42}), path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            restored, _, _ = bt.create_model("resnet18", pretrained=False)
            restored.load_state_dict(checkpoint["state_dict"])
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(sample), restored(sample))


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_wilson(self):
        result = bt.compute_metrics([0, 1, 2], [0, 0, 2])
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        lower, upper = bt.wilson_interval(2, 3)
        self.assertLess(lower, 2 / 3)
        self.assertGreater(upper, 2 / 3)

    def test_winner_tie_breaks_and_missing_results(self):
        results = [
            {"model": name, "variant": variant, "validation": {"accuracy": 0.8, "macro_f1": 0.7}}
            for name in bt.MODEL_NAMES for variant in ("frozen", "fine_tuned")
        ]
        self.assertEqual(bt.choose_winner(results)["model"], "efficientnet_b0")
        self.assertEqual(bt.choose_winner(results)["variant"], "frozen")
        results[-1]["validation"]["macro_f1"] = 0.8
        self.assertEqual(bt.choose_winner(results)["variant"], "fine_tuned")
        self.assertEqual(bt.choose_winner(results)["model"], "mobilenet_v3_large")
        with self.assertRaisesRegex(ValueError, "six"):
            bt.choose_winner(results[:-1])

    def test_test_split_access_only_after_selection(self):
        data = bt.load_metadata(bt.DEFAULT_DATA)
        valid = bt.BeansDataset(data, 1, bt.preprocessing("resnet18"))
        train = bt.BeansDataset(data, 0, bt.preprocessing("resnet18"))
        self.assertTrue(set(train.indices).isdisjoint(set(valid.indices)))
        self.assertTrue(np.all(data[3][train.indices] == 0))
        self.assertTrue(np.all(data[3][valid.indices] == 1))

    def test_orchestration_selects_before_test_evaluation(self):
        dummy_model = nn.Linear(3, 3)
        metrics = bt.compute_metrics([0, 1, 2], [0, 1, 2])
        metrics["loss"] = 0.1
        states = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = argparse.Namespace(dataset=bt.DEFAULT_DATA, output=output, epochs=1,
                                      patience=1, batch_size=2, threads=1, seed=42)

            def fake_train(name, variant, data, config, destination, initial_state=None):
                self.assertFalse((destination / "validation_results.json").exists())
                checkpoint = f"{name}_{variant}.pt"
                torch.save({"state_dict": dummy_model.state_dict()}, destination / checkpoint)
                states.append((name, variant, initial_state is not None))
                return {"model": name, "variant": variant, "checkpoint": checkpoint,
                        "validation": metrics}, dummy_model.state_dict()

            def fake_evaluate(model, loader):
                self.assertTrue((output / "validation_results.json").is_file())
                self.assertTrue(np.all(data[3][loader.dataset.indices] == 2))
                return dict(metrics)

            data = bt.load_metadata(bt.DEFAULT_DATA)
            with mock.patch.object(bt, "train_experiment", side_effect=fake_train), \
                 mock.patch.object(bt, "create_model", return_value=(dummy_model, None, None)), \
                 mock.patch.object(bt, "evaluate", side_effect=fake_evaluate) as evaluation:
                bt.run(args)
            self.assertEqual(len(states), 6)
            self.assertTrue(all(initial == (variant == "fine_tuned") for _, variant, initial in states))
            self.assertEqual(evaluation.call_count, 1)
            with (output / "validation_results.json").open() as file:
                selection = json.load(file)["winner"]
            with (output / "test_results.json").open() as file:
                report = json.load(file)
            self.assertEqual(report["winner"], selection)


if __name__ == "__main__":
    unittest.main()
