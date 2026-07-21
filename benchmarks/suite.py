"""Benchmark suite: diverse classification datasets for RYS evaluation.

Real-world datasets:
  - ImageNet-1K:    Object classification (1000 classes) — manual download required
  - DTD:            Texture recognition (47 classes)
  - EuroSAT:        Satellite land use (10 classes)
  - Places365:      Scene recognition (365 classes)
  - Stanford 40:    Action recognition (40 classes)
  - FGVC Aircraft:  Viewpoint / 3D structure (100 classes)

Synthetic reasoning probes (no download needed):
  - Counting:         Numerosity perception (10 classes)
  - Same/Different:   Relational identity (2 classes)
  - Spatial Relation: Spatial reasoning (4 classes)
  - Symmetry:         Global structure perception (2 classes)
  - Inside/Outside:   Topological reasoning (2 classes)
"""

import os
import tarfile
import urllib.request

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

from benchmarks.synthetic import (
    CountingDataset,
    InsideOutsideDataset,
    SameDifferentDataset,
    SpatialRelationDataset,
    SymmetryDataset,
)


# ── Custom dataset classes ──


class HFImageNet(Dataset):
    """ImageNet-1K via HuggingFace datasets (auto-download).

    Requires: pip install datasets
    One-time setup: accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
    then run: huggingface-cli login
    """

    def __init__(self, split="train", transform=None):
        if hf_load_dataset is None:
            raise ImportError(
                "Install the `datasets` package: pip install datasets>=2.14"
            )
        hf_split = "validation" if split == "val" else split
        self._ds = hf_load_dataset(
            "ILSVRC/imagenet-1k", split=hf_split, trust_remote_code=True
        )
        self.transform = transform
        self.targets = self._ds["label"]
        self.classes = self._ds.features["label"].names

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        item = self._ds[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


class Stanford40(Dataset):
    """Stanford 40 Actions dataset.

    http://vision.stanford.edu/Datasets/40actions.html
    """

    # NOTE: the historical .tar URLs now return 404; Stanford serves .zip.
    URLS = {
        "images": "http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip",
        "splits": "http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip",
    }

    def __init__(self, root, split="train", transform=None, download=False):
        self.root = root
        self.split = split
        self.transform = transform

        if download:
            self._download()

        self._load_data()

    def _download(self):
        import zipfile

        os.makedirs(self.root, exist_ok=True)
        for name, url in self.URLS.items():
            archive_path = os.path.join(self.root, os.path.basename(url))
            target_dir = os.path.join(
                self.root,
                "JPEGImages" if name == "images" else "ImageSplits",
            )
            if os.path.isdir(target_dir):
                continue
            if not os.path.exists(archive_path):
                print(f"    Downloading Stanford40 {name}...")
                urllib.request.urlretrieve(url, archive_path)
            print(f"    Extracting {name}...")
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(self.root)
            else:
                with tarfile.open(archive_path) as tar:
                    if hasattr(tarfile, "data_filter"):
                        tar.extractall(self.root, filter="data")
                    else:
                        tar.extractall(self.root)

    def _load_data(self):
        splits_dir = os.path.join(self.root, "ImageSplits")
        if not os.path.isdir(splits_dir):
            raise RuntimeError(
                f"Stanford 40 not found at {self.root}. "
                "Download from http://vision.stanford.edu/Datasets/40actions.html"
            )

        # Discover action names from split files
        suffix = f"_{self.split}.txt"
        action_names = sorted(
            fname[: -len(suffix)]
            for fname in os.listdir(splits_dir)
            if fname.endswith(suffix)
        )
        action_to_idx = {a: i for i, a in enumerate(action_names)}

        self.images = []
        self.targets = []
        self.classes = action_names

        for action in action_names:
            split_file = os.path.join(splits_dir, f"{action}_{self.split}.txt")
            with open(split_file) as f:
                for line in f:
                    filename = line.strip()
                    if filename:
                        self.images.append(filename)
                        self.targets.append(action_to_idx[action])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]


# ── Dataset loaders ──
# Each returns (train_dataset, test_dataset)


def _load_imagenet(root, transform):
    """ImageNet-1K via HuggingFace datasets (auto-download).

    First time: accept terms at huggingface.co/datasets/ILSVRC/imagenet-1k
    and run `huggingface-cli login`.
    """
    train = HFImageNet(split="train", transform=transform)
    val = HFImageNet(split="val", transform=transform)
    return train, val


def _load_dtd(root, transform):
    train = datasets.DTD(root, split="train", download=True, transform=transform)
    val = datasets.DTD(root, split="val", download=True, transform=transform)
    test = datasets.DTD(root, split="test", download=True, transform=transform)
    return ConcatDataset([train, val]), test


def _ensure_eurosat(root):
    """Pre-seed EuroSAT if the default mirror (madm.dfki.de) is down.

    torchvision expects {root}/eurosat/2750/<class dirs>. Falls back to the
    official Zenodo record, whose zip extracts to EuroSAT_RGB/.
    """
    base = os.path.join(root, "eurosat")
    if os.path.isdir(os.path.join(base, "2750")):
        return
    try:
        # Quick reachability check on the torchvision mirror
        req = urllib.request.Request(
            "https://madm.dfki.de/files/sentinel/EuroSAT.zip", method="HEAD"
        )
        urllib.request.urlopen(req, timeout=15)
        return  # mirror is up; let torchvision handle download + checksum
    except Exception:
        pass

    import zipfile

    print("    madm.dfki.de unreachable — fetching EuroSAT from Zenodo...")
    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, "EuroSAT_RGB.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(
            "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip",
            zip_path,
        )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(base)
    src = os.path.join(base, "EuroSAT_RGB")
    if os.path.isdir(src) and not os.path.isdir(os.path.join(base, "2750")):
        os.rename(src, os.path.join(base, "2750"))


def _load_eurosat(root, transform):
    _ensure_eurosat(root)
    full = datasets.EuroSAT(root, download=True, transform=transform)
    n = len(full)
    n_train = int(0.8 * n)
    n_test = n - n_train
    gen = torch.Generator().manual_seed(42)
    train, test = torch.utils.data.random_split(full, [n_train, n_test], generator=gen)
    return train, test


def _load_places365(root, transform):
    train = datasets.Places365(
        root, split="train-standard", small=True,
        download=True, transform=transform,
    )
    val = datasets.Places365(
        root, split="val", small=True,
        download=True, transform=transform,
    )
    return train, val


def _load_stanford40(root, transform):
    train = Stanford40(root, split="train", transform=transform, download=True)
    test = Stanford40(root, split="test", transform=transform, download=True)
    return train, test


def _load_fgvc_aircraft(root, transform):
    train = datasets.FGVCAircraft(
        root, split="trainval", download=True, transform=transform,
    )
    test = datasets.FGVCAircraft(
        root, split="test", download=True, transform=transform,
    )
    return train, test


# ── Synthetic probe loaders ──
# These ignore the root directory — images are generated procedurally.
# Different seeds for train/test ensure no overlap.


def _load_counting(root, transform):
    return (
        CountingDataset(n_per_class=500, transform=transform, seed=42),
        CountingDataset(n_per_class=200, transform=transform, seed=123),
    )


def _load_same_different(root, transform):
    return (
        SameDifferentDataset(n_per_class=2500, transform=transform, seed=42),
        SameDifferentDataset(n_per_class=1000, transform=transform, seed=123),
    )


def _load_spatial_relation(root, transform):
    return (
        SpatialRelationDataset(n_per_class=1250, transform=transform, seed=42),
        SpatialRelationDataset(n_per_class=500, transform=transform, seed=123),
    )


def _load_symmetry(root, transform):
    return (
        SymmetryDataset(n_per_class=2500, transform=transform, seed=42),
        SymmetryDataset(n_per_class=1000, transform=transform, seed=123),
    )


def _load_inside_outside(root, transform):
    return (
        InsideOutsideDataset(n_per_class=2500, transform=transform, seed=42),
        InsideOutsideDataset(n_per_class=1000, transform=transform, seed=123),
    )


BENCHMARK_DATASETS = [
    # Real-world datasets
    ("imagenet", _load_imagenet),
    ("dtd", _load_dtd),
    ("eurosat", _load_eurosat),
    ("places365", _load_places365),
    ("stanford40", _load_stanford40),
    ("fgvc_aircraft", _load_fgvc_aircraft),
    # Synthetic reasoning probes
    ("counting", _load_counting),
    ("same_different", _load_same_different),
    ("spatial_relation", _load_spatial_relation),
    ("symmetry", _load_symmetry),
    ("inside_outside", _load_inside_outside),
]


def get_transform(processor, image_size=224):
    """Build a torchvision transform matching the model's expected input size."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


# ── Label extraction ──

def get_labels(dataset) -> np.ndarray:
    """Extract labels from a dataset without loading images."""
    if isinstance(dataset, Subset):
        parent_labels = get_labels(dataset.dataset)
        indices = dataset.indices
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return parent_labels[indices]
    elif isinstance(dataset, ConcatDataset):
        return np.concatenate([get_labels(ds) for ds in dataset.datasets])
    elif hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    elif hasattr(dataset, "_labels"):
        return np.array(dataset._labels)
    elif hasattr(dataset, "imgs"):
        return np.array([target for _, target in dataset.imgs])
    else:
        # Fallback: iterate (slow but universal)
        return np.array([label for _, label in dataset])


# ── Subset selection ──

def choose_class_subset(labels: np.ndarray, max_classes: int, seed: int = 42):
    """Deterministically choose up to max_classes class ids (0 = all)."""
    classes = np.unique(labels)
    if max_classes and len(classes) > max_classes:
        rng = np.random.default_rng(seed)
        classes = np.sort(rng.choice(classes, max_classes, replace=False))
    return classes


def select_reference_subset(
    dataset, n_per_class: int = 5, seed: int = 42, class_subset=None
) -> tuple[Subset, np.ndarray]:
    """Select n_per_class images per class from a dataset.

    Args:
        class_subset: optional array of class ids to restrict to.

    Returns:
        (Subset, labels array) for the selected reference images.
    """
    labels = get_labels(dataset)
    classes = np.unique(labels) if class_subset is None else np.asarray(class_subset)
    rng = np.random.default_rng(seed)
    indices = []
    for label in classes:
        class_indices = np.where(labels == label)[0]
        n = min(n_per_class, len(class_indices))
        selected = rng.choice(class_indices, n, replace=False)
        indices.extend(selected.tolist())
    return Subset(dataset, indices), labels[indices]


def select_candidate_subset(
    dataset, n_candidates: int = 1000, seed: int = 42, class_subset=None
) -> tuple[Subset, np.ndarray]:
    """Select a random subset as candidates for probe-image selection.

    Args:
        class_subset: optional array of class ids to restrict to (keeps the
        candidate pool consistent with the reference centroids).

    Returns:
        (Subset, labels array) for the selected candidates.
    """
    labels = get_labels(dataset)
    if class_subset is not None:
        pool = np.where(np.isin(labels, np.asarray(class_subset)))[0]
    else:
        pool = np.arange(len(labels))
    n = min(n_candidates, len(pool))
    rng = np.random.default_rng(seed)
    indices = rng.choice(pool, n, replace=False).tolist()
    return Subset(dataset, indices), labels[indices]


def load_benchmark(
    data_root: str,
    processor,
    dataset_names: list[str] | None = None,
    image_size: int = 224,
    allow_missing: bool = False,
):
    """Load the benchmark suite.

    Args:
        data_root: Root directory for dataset downloads.
        processor: CLIPImageProcessor for building transforms.
        dataset_names: Subset of dataset names to load (None = all).
        image_size: Input resolution expected by the model.
        allow_missing: if False (default), any dataset that fails to load
            raises — a silently shrinking benchmark would otherwise produce
            aggregates that look complete but are not.

    Returns:
        dict of {name: (train_dataset, test_dataset)}.
    """
    transform = get_transform(processor, image_size=image_size)
    benchmark = {}
    failures = {}

    for name, loader_fn in BENCHMARK_DATASETS:
        if dataset_names is not None and name not in dataset_names:
            continue

        print(f"  Loading {name}...", end=" ", flush=True)
        try:
            train_ds, test_ds = loader_fn(
                os.path.join(data_root, name), transform
            )
            benchmark[name] = (train_ds, test_ds)
            print(f"train={len(train_ds)}, test={len(test_ds)}")
        except Exception as e:
            failures[name] = repr(e)
            print(f"FAILED: {e}")

    if failures and not allow_missing:
        raise RuntimeError(
            f"Datasets failed to load: {failures}. "
            "Fix them or pass --allow-missing-datasets to proceed without."
        )

    return benchmark
