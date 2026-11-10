"""Reproducible CPU transfer-learning experiments for the beans dataset."""

import argparse
import copy
import hashlib
import io
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import PIL
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


CLASS_NAMES = ("angular_leaf_spot", "bean_rust", "healthy")
SPLIT_NAMES = ("train", "validation", "test")
MODEL_NAMES = ("efficientnet_b0", "resnet18", "mobilenet_v3_large")
WEIGHTS = {
    "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
}
DEFAULT_DATA = Path(__file__).resolve().parents[3] / "session2_inside_the_black_box/src/data/beans128.npz"


def load_metadata(path):
    with np.load(path, allow_pickle=False) as pack:
        jpeg, offsets, labels, splits = (pack[key] for key in ("jpeg", "offsets", "labels", "split"))
    if jpeg.ndim != 1 or jpeg.dtype != np.uint8:
        raise ValueError("jpeg must be a one-dimensional uint8 array")
    for name, array in (("offsets", offsets), ("labels", labels), ("split", splits)):
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
    if not len(labels) or len(splits) != len(labels) or len(offsets) != len(labels) + 1:
        raise ValueError("Inconsistent dataset array lengths")
    if offsets[0] != 0 or offsets[-1] != len(jpeg) or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("Invalid JPEG offsets")
    if not np.isin(labels, range(3)).all() or not np.isin(splits, range(3)).all():
        raise ValueError("Unknown class or split value")
    counts = [[int(np.sum((labels == c) & (splits == s))) for c in range(3)] for s in range(3)]
    if any(count == 0 for row in counts for count in row):
        raise ValueError("All three classes must be represented in each split")
    return jpeg, offsets, labels, splits, counts


class BeansDataset(Dataset):
    def __init__(self, data, split_id, transform):
        self.jpeg, self.offsets, self.labels, splits, _ = data
        self.indices = np.flatnonzero(splits == split_id)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        image_bytes = self.jpeg[self.offsets[index]:self.offsets[index + 1]].tobytes()
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                if image.size != (128, 128):
                    raise ValueError(f"Unexpected image size {image.size}")
                image = image.convert("RGB")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not decode dataset image {index}") from exc
        return self.transform(image), int(self.labels[index])


def preprocessing(model_name, training=False):
    weights = WEIGHTS[model_name]
    inference = weights.transforms()
    if not training:
        return inference
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.1),
        inference,
    ])


def create_model(name, pretrained=True):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    model = getattr(models, name)(weights=WEIGHTS[name] if pretrained else None)
    if name == "resnet18":
        head = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.fc = head
        final_block = model.layer4
    else:
        old_head = model.classifier[-1]
        head = nn.Linear(old_head.in_features, len(CLASS_NAMES))
        model.classifier[-1] = head
        final_block = model.features[-1]
    return model, head, final_block


def set_trainable(model, head, final_block, fine_tune):
    for parameter in model.parameters():
        parameter.requires_grad = False
    if fine_tune:
        for parameter in final_block.parameters():
            parameter.requires_grad = True
    for parameter in head.parameters():
        parameter.requires_grad = True
    # Keep batch-normalization parameters and running statistics fixed.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in module.parameters():
                parameter.requires_grad = False


def training_mode(model, head, final_block, fine_tune):
    model.eval()
    head.train()
    if fine_tune:
        final_block.train()
        for module in final_block.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def compute_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(3)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(3)), average="macro", zero_division=0)),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(3))).tolist(),
    }


def evaluate(model, loader):
    model.eval()
    true, predicted = [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            true.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).tolist())
    if not true:
        raise ValueError("Cannot evaluate an empty dataset")
    metrics = compute_metrics(true, predicted)
    metrics["loss"] = loss_sum / len(true)
    return metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def checkpoint_payload(model, model_name, variant, config):
    return {
        "model": model_name,
        "variant": variant,
        "weights": str(WEIGHTS[model_name]),
        "class_names": list(CLASS_NAMES),
        "config": config,
        "preprocessing": {"train": str(preprocessing(model_name, True)),
                          "evaluation": str(preprocessing(model_name))},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__), "numpy": str(np.__version__),
                     "pillow": str(PIL.__version__), "sklearn": str(sklearn.__version__)},
        "state_dict": copy.deepcopy(model.state_dict()),
    }


def train_experiment(name, variant, data, config, output, initial_state=None):
    seed_everything(config["seed"])
    fine_tune = variant == "fine_tuned"
    model, head, final_block = create_model(name, pretrained=initial_state is None)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    set_trainable(model, head, final_block, fine_tune)
    head_ids = {id(parameter) for parameter in head.parameters()}
    backbone_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": list(head.parameters()), "lr": config["head_lr"]}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config["backbone_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        BeansDataset(data, 0, preprocessing(name, training=True)), batch_size=config["batch_size"],
        shuffle=True, generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        BeansDataset(data, 1, preprocessing(name)), batch_size=config["batch_size"], shuffle=False, num_workers=0,
    )
    best_key = (-1.0, -float("inf"))
    best_state = None
    best_metrics = None
    best_epoch = 0
    history = []
    stale_epochs = 0
    started = time.monotonic()
    for epoch in range(1, config["epochs"] + 1):
        training_mode(model, head, final_block, fine_tune)
        train_loss, train_count = 0.0, 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_count += len(labels)
        metrics = evaluate(model, validation_loader)
        history.append({"epoch": epoch, "train_loss": train_loss / train_count,
                        "validation_accuracy": metrics["accuracy"], "validation_loss": metrics["loss"]})
        print(f"{name} {variant}: epoch {epoch}, train loss {train_loss / train_count:.4f}, "
              f"validation accuracy {metrics['accuracy']:.4f}, loss {metrics['loss']:.4f}", flush=True)
        key = (metrics["accuracy"], -metrics["loss"])
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                break
    elapsed = time.monotonic() - started
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint produced for {name} {variant}")
    model.load_state_dict(best_state)
    checkpoint_path = output / f"{name}_{variant}.pt"
    torch.save(checkpoint_payload(model, name, variant, config), checkpoint_path)
    result = {
        "model": name, "variant": variant, "best_epoch": best_epoch,
        "validation": best_metrics, "training_seconds": elapsed,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint": checkpoint_path.name, "history": history,
    }
    save_json(output / f"{name}_{variant}.json", result)
    return result, best_state


def choose_winner(results):
    expected = {(name, variant) for name in MODEL_NAMES for variant in ("frozen", "fine_tuned")}
    if {(r["model"], r["variant"]) for r in results} != expected or len(results) != len(expected):
        raise ValueError("Winner selection requires exactly the six planned experiments")
    order = {name: index for index, name in enumerate(MODEL_NAMES)}
    return max(results, key=lambda r: (
        r["validation"]["accuracy"], r["validation"]["macro_f1"],
        r["variant"] == "frozen", -order[r["model"]],
    ))


def wilson_interval(successes, total):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial count")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, patience, batch size and threads must be positive")
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    data = load_metadata(dataset_path)
    config = {
        "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "threads": args.threads, "head_lr": 0.001,
        "backbone_lr": 0.0001, "weight_decay": 0.01,
    }
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    save_json(output / "run_config.json", {
        "config": config, "dataset": str(dataset_path), "dataset_sha256": digest,
        "split_class_counts": dict(zip(SPLIT_NAMES, data[4])), "class_names": CLASS_NAMES,
        "models": {name: str(WEIGHTS[name]) for name in MODEL_NAMES},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__,
                     "pillow": PIL.__version__, "sklearn": sklearn.__version__},
        "preprocessing": {name: {"train": str(preprocessing(name, True)), "evaluation": str(preprocessing(name))}
                          for name in MODEL_NAMES},
    })
    results = []
    for name in MODEL_NAMES:
        frozen, frozen_state = train_experiment(name, "frozen", data, config, output)
        results.append(frozen)
        fine_tuned, _ = train_experiment(name, "fine_tuned", data, config, output, frozen_state)
        results.append(fine_tuned)
    train_labels = data[2][data[3] == 0]
    val_labels = data[2][data[3] == 1]
    majority = int(np.bincount(train_labels, minlength=3).argmax())
    majority_metrics = compute_metrics(val_labels, np.full(len(val_labels), majority))
    winner = choose_winner(results)
    save_json(output / "validation_results.json", {
        "selection_rule": "accuracy, macro F1, frozen preference, model order: " + ", ".join(MODEL_NAMES),
        "majority_class": CLASS_NAMES[majority], "majority_baseline": majority_metrics,
        "experiments": results, "winner": {"model": winner["model"], "variant": winner["variant"],
                                            "checkpoint": winner["checkpoint"]},
    })
    print(f"Selected from validation: {winner['model']} {winner['variant']}", flush=True)
    checkpoint = torch.load(output / winner["checkpoint"], map_location="cpu", weights_only=True)
    model, _, _ = create_model(winner["model"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loader = DataLoader(
        BeansDataset(data, 2, preprocessing(winner["model"])), batch_size=args.batch_size, shuffle=False,
    )
    test_metrics = evaluate(model, test_loader)
    test_count = sum(item["support"] for item in test_metrics["per_class"].values())
    correct = sum(test_metrics["confusion_matrix"][i][i] for i in range(3))
    test_metrics["accuracy_95_percent_wilson_interval"] = wilson_interval(correct, test_count)
    save_json(output / "test_results.json", {
        "winner": {"model": winner["model"], "variant": winner["variant"], "checkpoint": winner["checkpoint"]},
        "test": test_metrics,
    })
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}, macro F1: {test_metrics['macro_f1']:.4f}", flush=True)


def predict(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    name = checkpoint["model"]
    if checkpoint["class_names"] != list(CLASS_NAMES) or checkpoint["weights"] != str(WEIGHTS[name]):
        raise ValueError("Checkpoint class order or weights preprocessing does not match this program")
    model, _, _ = create_model(name, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with Image.open(args.image) as image:
        image.load()
        tensor = preprocessing(name)(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].tolist()
    print(json.dumps({"predicted_class": CLASS_NAMES[int(np.argmax(probabilities))],
                      "class_probabilities": dict(zip(CLASS_NAMES, probabilities))}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run all six experiments and evaluate the winner")
    train.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    train.add_argument("--output", type=Path, required=True, help="New directory for outputs; must not exist")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    inference = commands.add_parser("predict", help="Classify a local image with a saved checkpoint")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        run(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
