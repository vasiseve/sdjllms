# ============================================
# Representation Holonomy — ICLR Experiments (Standalone, Colab-safe)
# ============================================
# Requires: torch, torchvision, matplotlib, pandas

from __future__ import annotations
import os, math, json, random, time, gc
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (saves RAM in Colab)
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 140

# -----------------------------
# Drive mount & results folder
# -----------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
    RESULTS_DIR = "/content/drive/MyDrive/holonomy_results"
except Exception:
    # fallback for non-Colab envs
    RESULTS_DIR = "./holonomy_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Utils: seeding, device, I/O
# -----------------------------
def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(1)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def free_run(run: "Run"):
    try:
        run.model.cpu()
        del run.model
        del run.train_loader, run.train_plain_loader, run.test_loader, run
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()


def _unpack_batch(b):
    # supports (x,y) and (x,y,i) and dicts with typical keys
    if isinstance(b, (list, tuple)):
        if len(b) == 3:
            return b[0], b[1]
        if len(b) == 2:
            return b[0], b[1]
    if isinstance(b, dict):
        x = b.get("x") or b.get("inputs")
        y = b.get("y") or b.get("labels")
        return x, y
    return b  # assume (x,y)


# -----------------------------
# Dataset wrappers & loaders
# -----------------------------
class IndexedTorchDataset(Dataset):
    """Wrap any torchvision dataset to also return the sample index."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i


def build_datasets(dataset: str, augment: bool = True):
    ds = dataset.lower()
    if ds == "mnist":
        mean, std = (0.1307,), (0.3081,)
        tf_aug = T.Compose([T.RandomRotation(10), T.ToTensor(), T.Normalize(mean, std)])
        tf_base = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        tr_aug = torchvision.datasets.MNIST(
            "./data", train=True, download=True, transform=(tf_aug if augment else tf_base)
        )
        tr_base = torchvision.datasets.MNIST(
            "./data", train=True, download=True, transform=tf_base
        )
        te = torchvision.datasets.MNIST(
            "./data", train=False, download=True, transform=tf_base
        )
        in_shape, n_classes = (1, 28, 28), 10
    elif ds == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tf_aug = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        tf_base = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        tr_aug = torchvision.datasets.CIFAR10(
            "./data", train=True, download=True, transform=(tf_aug if augment else tf_base)
        )
        tr_base = torchvision.datasets.CIFAR10(
            "./data", train=True, download=True, transform=tf_base
        )
        te = torchvision.datasets.CIFAR10(
            "./data", train=False, download=True, transform=tf_base
        )
        in_shape, n_classes = (3, 32, 32), 10
    elif ds == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        tf_aug = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        tf_base = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        tr_aug = torchvision.datasets.CIFAR100(
            "./data", train=True, download=True, transform=(tf_aug if augment else tf_base)
        )
        tr_base = torchvision.datasets.CIFAR100(
            "./data", train=True, download=True, transform=tf_base
        )
        te = torchvision.datasets.CIFAR100(
            "./data", train=False, download=True, transform=tf_base
        )
        in_shape, n_classes = (3, 32, 32), 100
    else:
        raise ValueError("dataset must be 'mnist' | 'cifar10' | 'cifar100'")
    return (IndexedTorchDataset(tr_aug), IndexedTorchDataset(tr_base), te), in_shape, n_classes


def build_loaders(train_aug, train_plain, test, batch_size=128, workers=None):
    if workers is None:
        workers = 4 if torch.cuda.is_available() else 0
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    train_plain_loader = DataLoader(
        train_plain,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    test_loader = DataLoader(
        test,
        batch_size=256,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    return train_loader, train_plain_loader, test_loader


# -----------------------------
# Models
# -----------------------------
class MNISTMLP(nn.Module):
    def __init__(self, hidden=512, num_classes=10, linearized: bool = False):
        super().__init__()
        act = nn.Identity if linearized else (lambda: nn.ReLU(inplace=True))
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Sequential(nn.Linear(28 * 28, hidden), act())
        self.hidden2 = nn.Sequential(nn.Linear(hidden, hidden), act())
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        z1 = self.hidden1(x)
        z2 = self.hidden2(z1)
        return self.head(z2)


def build_resnet18_cifar(num_classes=10):
    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_model(dataset: str, model_name: str, num_classes: int, linearized: bool = False):
    ds, name = dataset.lower(), model_name.lower()
    if ds == "mnist":
        return MNISTMLP(hidden=512, num_classes=num_classes, linearized=linearized).to(device)
    if ds in ("cifar10", "cifar100"):
        if name == "resnet18":
            return build_resnet18_cifar(num_classes).to(device)
        else:
            raise ValueError("For CIFAR use model_name='resnet18'")
    raise ValueError


# -----------------------------
# Training regimes (ERM / LS / mixup / adv)
# -----------------------------
def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_m, (y_a, y_b, lam), idx


def mixup_criterion(criterion, pred, y_mix):
    y_a, y_b, lam = y_mix
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fgsm(model, x, y, eps=4 / 255):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    x_adv = (x + eps * x.grad.sign()).clamp(0, 1)
    return x_adv.detach()


def pgd_adversary(model, x, y, eps=4 / 255, step=2 / 255, iters=3):
    x0 = x.detach()
    x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1).detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + step * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
            x_adv = x_adv.clamp(0, 1).detach()
    return x_adv


@dataclass
class Regime:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    adv_train: bool = False
    pgd_eps: float = 4 / 255
    pgd_step: float = 2 / 255
    pgd_iters: int = 3


@dataclass
class Run:
    model: nn.Module
    train_loader: DataLoader
    train_plain_loader: DataLoader
    test_loader: DataLoader
    cfg: Dict[str, Any]


def train_run(
    seed: int,
    dataset: str = "mnist",
    model_name: str = "mnist_mlp",
    regime: Regime = Regime(),
    linearized: bool = False,
) -> Run:
    set_seed(seed)
    (tr_aug, tr_plain, te), in_shape, n_classes = build_datasets(dataset, augment=True)
    train_loader, train_plain_loader, test_loader = build_loaders(
        tr_aug, tr_plain, te, batch_size=regime.batch_size
    )
    model = build_model(dataset, model_name, n_classes, linearized=linearized)
    opt = torch.optim.Adam(model.parameters(), lr=regime.lr, weight_decay=regime.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=regime.label_smoothing)

    for ep in range(regime.epochs):
        model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if regime.mixup_alpha > 0:
                xb_in, y_mix, _ = mixup_batch(xb, yb, alpha=regime.mixup_alpha)
                logits = model(xb_in)
                loss = mixup_criterion(criterion, logits, y_mix)
            else:
                xb_in = xb
                if regime.adv_train:
                    xb_in = pgd_adversary(
                        model,
                        xb_in,
                        yb,
                        eps=regime.pgd_eps,
                        step=regime.pgd_step,
                        iters=regime.pgd_iters,
                    )
                logits = model(xb_in)
                loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return Run(
        model=model,
        train_loader=train_loader,
        train_plain_loader=train_plain_loader,
        test_loader=test_loader,
        cfg=dict(seed=seed, dataset=dataset, model=model_name, linearized=linearized, **regime.__dict__),
    )


# -----------------------------
# Representation compression (conv layers) + capture
# -----------------------------
COMPRESS_REP = True
POOL_HW = 1
DIM_RP = 1024
_RP_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


def _get_proj(in_dim: int, out_dim: int, device: torch.device) -> torch.Tensor:
    key = (in_dim, out_dim)
    if key not in _RP_CACHE:
        g = torch.randn(in_dim, out_dim, device=device) / (out_dim ** 0.5)
        q, _ = torch.linalg.qr(g, mode="reduced")
        if q.size(1) > out_dim:
            q = q[:, :out_dim]
        _RP_CACHE[key] = q.detach().cpu()
    return _RP_CACHE[key].to(device)


def _postprocess_rep_tensor(z: torch.Tensor) -> torch.Tensor:
    if not COMPRESS_REP:
        return z.flatten(1)
    if z.ndim == 4:
        if POOL_HW == 1:
            z = z.mean(dim=(2, 3), keepdim=False)
        else:
            z = F.adaptive_avg_pool2d(z, (POOL_HW, POOL_HW))
            z = z.flatten(1)
    else:
        z = z.flatten(1)
    if DIM_RP is not None and z.size(1) > DIM_RP:
        P = _get_proj(z.size(1), DIM_RP, z.device)
        z = z @ P
    return z


class ActivationCatcher:
    """Capture activations from a given layer via a forward hook, with compression."""
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.storage: Dict[str, torch.Tensor] = {}

        def _hook(_m, _inp, out):
            self.storage["z"] = _postprocess_rep_tensor(out.detach())

        self.handle = layer.register_forward_hook(_hook)
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.model(x)
        return self.storage["z"]

    def close(self):
        self.handle.remove()


@torch.no_grad()
def rep_of(model: nn.Module, layer: nn.Module, x: torch.Tensor) -> np.ndarray:
    c = ActivationCatcher(model, layer)
    z = c(x).detach().cpu().numpy()
    c.close()
    return z


@torch.no_grad()
def collect_rep_pool_Zonly(
    model: nn.Module,
    layer: nn.Module,
    neighbor_loader: Iterable,
    max_samples: int = 2048,
) -> np.ndarray:
    c = ActivationCatcher(model, layer)
    Zs, total = [], 0
    for xb, yb, _ in neighbor_loader:
        xb = xb.to(next(model.parameters()).device)
        z = c(xb)
        Zs.append(z.cpu().numpy())
        total += xb.size(0)
        if total >= max_samples:
            break
    c.close()
    return np.concatenate(Zs, 0)


# -----------------------------
# Loop construction (local PCA in input space)
# -----------------------------
def _strip_batch(x: torch.Tensor) -> torch.Tensor:
    return x[0] if x.dim() >= 2 and x.size(0) == 1 else x


def make_pca_loop_around(
    x0: torch.Tensor,
    neighbor_loader: Iterable,
    k_neighbors: int = 512,
    radius: float = 0.1,
    n_points: int = 12,
) -> torch.Tensor:
    dev = x0.device
    x0_nob = _strip_batch(x0)
    x0f = x0_nob.view(1, -1)

    Xpool = []
    for xb, yb, _ in neighbor_loader:
        Xpool.append(xb)
        if sum(t.size(0) for t in Xpool) >= max(2048, 4 * k_neighbors):
            break
    X = torch.cat(Xpool, 0).to(dev)
    Xf = X.view(X.size(0), -1)

    d = torch.norm(Xf - x0f, dim=1)
    k = min(k_neighbors, X.size(0))
    idx = torch.topk(d, k=k, largest=False).indices
    Xn = Xf[idx]

    Xc = Xn - Xn.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:2]

    th = torch.linspace(0, 2 * math.pi, n_points + 1, device=dev)[:-1]
    circle = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
    displacements = (circle @ basis) * radius

    pts = x0f + displacements
    return pts.view(n_points, *x0_nob.shape)


# -----------------------------
# Holonomy (SO(p) subspace Procrustes, shared neighbors, global whitening)
# -----------------------------
@dataclass
class HolonomyRepResult:
    H: np.ndarray
    fro: float
    h_norm: float
    eig_angles: np.ndarray
    settings: Dict[str, Any]


def _orthonormal_basis(Z: np.ndarray, q: int) -> np.ndarray:
    Zc = Z - Z.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    q = min(q, Vt.shape[0])
    return Vt[:q].T


def _weighted_center(Z: np.ndarray, z_center: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(Z - z_center[None, :], axis=1)
    sigma = np.median(d) + 1e-6
    w = np.exp(-d / sigma)
    w = w / (w.sum() + 1e-12)
    return (w[:, None] * Z).sum(0, keepdims=True)


def _so_procrustes(Zs_src: np.ndarray, Zs_tgt: np.ndarray, q: int) -> np.ndarray:
    k, p = Zs_src.shape
    q = max(1, min(q, p, k - 1))
    B = _orthonormal_basis(np.vstack([Zs_src, Zs_tgt]), q=q)
    X = Zs_src @ B
    Y = Zs_tgt @ B
    C = X.T @ Y
    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    Rq = U @ Vt
    if np.linalg.det(Rq) < 0:
        U[:, -1] *= -1
        Rq = U @ Vt
    return B @ Rq @ B.T + (np.eye(p) - B @ B.T)


def holonomy_rep_SO_procrustes_midpoint(
    model: nn.Module,
    layer: nn.Module,
    loop_points: torch.Tensor,
    neighbor_loader: Iterable,
    neighborhood_size: int = 96,
    q_subspace: int = 64,
    max_pool: int = 2048,
) -> HolonomyRepResult:
    Z_pool = collect_rep_pool_Zonly(model, layer, neighbor_loader, max_samples=max_pool)
    Zmu, Zsd = Z_pool.mean(0, keepdims=True), Z_pool.std(0, keepdims=True) + 1e-6
    Zw = (Z_pool - Zmu) / Zsd
    p = Zw.shape[1]
    H = np.eye(p, dtype=np.float32)

    def _kneighbors_rep_mid(zi_w: np.ndarray, zj_w: np.ndarray, k: int) -> np.ndarray:
        z_mid = 0.5 * (zi_w + zj_w)
        d = np.linalg.norm(Zw - z_mid[None, :], axis=1)
        k = min(k, d.shape[0])
        idx = np.argpartition(d, kth=k - 1)[:k]
        return idx[np.argsort(d[idx])]

    dev = next(model.parameters()).device
    for i in range(loop_points.size(0)):
        j = (i + 1) % loop_points.size(0)
        zi = rep_of(model, layer, loop_points[i : i + 1].to(dev))[0]
        zj = rep_of(model, layer, loop_points[j : j + 1].to(dev))[0]
        zi_w = (zi - Zmu.squeeze(0)) / Zsd.squeeze(0)
        zj_w = (zj - Zmu.squeeze(0)) / Zsd.squeeze(0)

        idx = _kneighbors_rep_mid(zi_w, zj_w, k=neighborhood_size)
        Zs = Zw[idx]
        mu_i = _weighted_center(Zs, zi_w)
        mu_j = _weighted_center(Zs, zj_w)
        Z_src = Zs - mu_i
        Z_tgt = Zs - mu_j

        Rij = _so_procrustes(Z_src, Z_tgt, q=q_subspace)
        H = Rij @ H

    HmI = H - np.eye(p, dtype=np.float32)
    fro = float(np.linalg.norm(HmI, ord="fro"))
    h_norm = float(fro / (2.0 * math.sqrt(p)))
    w, _ = np.linalg.eig(H.astype(np.complex128))
    ang = np.angle(w)
    return HolonomyRepResult(
        H=H,
        fro=fro,
        h_norm=h_norm,
        eig_angles=ang,
        settings=dict(neighborhood_size=neighborhood_size, q_subspace=q_subspace, pooled=POOL_HW, rp=DIM_RP),
    )


# -----------------------------
# Holonomy sweeps, ablations, k/q, training dynamics
# -----------------------------
def layer_by_name(model, name: str):
    if hasattr(model, name):
        return getattr(model, name)
    if name == "layer1":
        return model.layer1
    if name == "layer2":
        return model.layer2
    if name == "layer3":
        return model.layer3
    raise ValueError(f"Unknown layer name {name}")


def build_loop(x0, loader, r: float, plane: str = "pca", n_points: int = 12):
    if plane == "pca":
        return make_pca_loop_around(x0, loader, k_neighbors=512, radius=float(r), n_points=n_points)

    base = _strip_batch(x0)
    flat = base.view(1, -1)

    u = torch.randn_like(flat)
    u = u / (u.norm() + 1e-12)
    v = torch.randn_like(flat)
    v = v - (v @ u.T) * u
    v = v / (v.norm() + 1e-12)

    th = torch.linspace(0, 2 * math.pi, steps=n_points + 1, device=base.device)[:-1]
    loop = flat + float(r) * (
        torch.cos(th).unsqueeze(1) * u + torch.sin(th).unsqueeze(1) * v
    )

    assert loop.numel() == int(n_points * flat.numel()), f"{loop.numel()} vs {n_points * flat.numel()}"
    return loop.view(n_points, *base.shape)


def holonomy_curve_for_run_layer(
    run: Run,
    layer_name: str,
    radii,
    k=96,
    q=64,
    plane="pca",
    n_points=12,
):
    model = run.model.eval()
    layer = layer_by_name(model, layer_name)
    batch = next(iter(run.test_loader))
    xb_te, yb_te = _unpack_batch(batch)
    x0 = xb_te[:1].to(next(model.parameters()).device)
    rows = []
    for r in radii:
        loop = build_loop(x0, run.train_plain_loader, r, plane=plane, n_points=n_points)
        hol = holonomy_rep_SO_procrustes_midpoint(
            model,
            layer,
            loop,
            run.train_plain_loader,
            neighborhood_size=k,
            q_subspace=q,
            max_pool=2048,
        )
        rows.append(
            dict(
                seed=run.cfg["seed"],
                dataset=run.cfg["dataset"],
                model=run.cfg["model"],
                regime=json.dumps(
                    {
                        kk: vv
                        for kk, vv in run.cfg.items()
                        if kk in ["label_smoothing", "mixup_alpha", "adv_train"]
                    }
                ),
                layer=layer_name,
                plane=plane,
                k=k,
                q=q,
                radius=float(r),
                h_norm=hol.h_norm,
                max_angle=float(np.max(np.abs(hol.eig_angles))),
            )
        )
    return rows


def training_dynamics_curve(
    run: Run,
    layer_name: str,
    radii=(0.1,),
    epochs=5,
    k=96,
    q=64,
):
    set_seed(run.cfg["seed"])
    (tr_aug, tr_plain, te), in_shape, n_classes = build_datasets(run.cfg["dataset"], augment=True)
    train_loader, train_plain_loader, test_loader = build_loaders(
        tr_aug, tr_plain, te, batch_size=run.cfg["batch_size"]
    )
    model = build_model(
        run.cfg["dataset"],
        run.cfg["model"],
        n_classes,
        linearized=run.cfg.get("linearized", False),
    )
    opt = torch.optim.Adam(model.parameters(), lr=run.cfg["lr"], weight_decay=run.cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=run.cfg["label_smoothing"])

    batch = next(iter(test_loader))
    xb_te, yb_te = _unpack_batch(batch)
    x0 = xb_te[:1].to(device)

    layer = layer_by_name(model, layer_name)
    rows = []
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            xb, yb = _unpack_batch(batch)
            xb, yb = xb.to(device), yb.to(device)
            if run.cfg["mixup_alpha"] > 0:
                xb_in, y_mix, _ = mixup_batch(xb, yb, alpha=run.cfg["mixup_alpha"])
                logits = model(xb_in)
                loss = mixup_criterion(criterion, logits, y_mix)
            else:
                xb_in = xb
                if run.cfg["adv_train"]:
                    xb_in = pgd_adversary(
                        model,
                        xb_in,
                        yb,
                        eps=run.cfg["pgd_eps"],
                        step=run.cfg["pgd_step"],
                        iters=run.cfg["pgd_iters"],
                    )
                logits = model(xb_in)
                loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        for r in radii:
            loop = build_loop(x0, train_plain_loader, r, plane="pca", n_points=12)
            hol = holonomy_rep_SO_procrustes_midpoint(
                model,
                layer,
                loop,
                train_plain_loader,
                neighborhood_size=k,
                q_subspace=q,
                max_pool=2048,
            )
            rows.append(dict(epoch=ep + 1, radius=float(r), h_norm=hol.h_norm))
    return rows


# -----------------------------
# Pointwise similarity: CKA after Procrustes
# -----------------------------
def collect_reps(run: Run, layer_name: str, max_batches=8):
    model = run.model.eval()
    layer = layer_by_name(model, layer_name)
    catcher = ActivationCatcher(model, layer)
    Zs = []
    with torch.no_grad():
        for bi, batch in enumerate(run.test_loader):
            xb, yb = _unpack_batch(batch)
            xb = xb.to(next(model.parameters()).device)
            z = catcher(xb).detach().cpu().numpy()
            Zs.append(z)
            if bi + 1 >= max_batches:
                break
    catcher.close()
    return np.concatenate(Zs, 0)


def procrustes_Q(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def linear_cka(X, Y, center=True):
    Xc = X - X.mean(0, keepdims=True) if center else X
    Yc = Y - Y.mean(0, keepdims=True) if center else Y
    K, L = Xc @ Xc.T, Yc @ Yc.T
    num = (K * L).sum()
    den = np.sqrt((K * K).sum() * (L * L).sum() + 1e-12)
    return float(num / den)


def cka_after_procrustes(runA: Run, runB: Run, layer_name: str, max_batches=8):
    ZA = collect_reps(runA, layer_name, max_batches=max_batches)
    ZB = collect_reps(runB, layer_name, max_batches=max_batches)
    Q = procrustes_Q(ZA, ZB)
    return dict(
        cka_after=linear_cka(ZA @ Q, ZB),
        fro_after=float(
            np.linalg.norm(ZA @ Q - ZB, ord="fro") / (np.linalg.norm(ZB, ord="fro") + 1e-12)
        ),
    )


# -----------------------------
# Robustness panel: FGSM & simple corruptions
# -----------------------------
@torch.no_grad()
def accuracy(model: nn.Module, loader) -> float:
    model.eval()
    tot = 0
    ok = 0
    for batch in loader:
        xb, yb = _unpack_batch(batch)
        xb, yb = xb.to(device), yb.to(device)
        ok += (model(xb).argmax(1) == yb).sum().item()
        tot += yb.numel()
    return ok / max(1, tot)


def fgsm_accuracy(model: nn.Module, loader, eps=4 / 255) -> float:
    model.eval()
    tot = 0
    ok = 0
    for batch in loader:
        xb, yb = _unpack_batch(batch)
        xb, yb = xb.to(device), yb.to(device)
        xb_adv = xb.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(xb_adv), yb)
        loss.backward()
        xb_adv = (xb_adv + eps * xb_adv.grad.sign()).clamp(0, 1).detach()
        ok += (model(xb_adv).argmax(1) == yb).sum().item()
        tot += yb.numel()
    return ok / max(1, tot)


def corrupted_loader(base_test, mean, std):
    ds_cls = type(base_test)
    root = getattr(base_test, "root", "./data")

    blur_tf = T.Compose(
        [
            T.GaussianBlur(3, sigma=(0.5, 1.0)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    jitter_tf = T.Compose(
        [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    def add_noise_tensor(t):
        return (t + 0.08 * torch.randn_like(t)).clamp(0, 1)

    noise_tf = T.Compose(
        [
            T.ToTensor(),
            T.Lambda(add_noise_tensor),
            T.Normalize(mean, std),
        ]
    )

    datasets = [
        ds_cls(root, train=False, download=True, transform=blur_tf),
        ds_cls(root, train=False, download=True, transform=jitter_tf),
        ds_cls(root, train=False, download=True, transform=noise_tf),
    ]

    def loaders():
        for d in datasets:
            yield DataLoader(d, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

    return loaders


def avg_corruption_accuracy(model, base_test, mean, std):
    accs = []
    for ld in corrupted_loader(base_test, mean, std)():
        accs.append(accuracy(model, ld))
    return float(np.mean(accs))


# -----------------------------
# Plotting & CSV helpers
# -----------------------------
def plot_radius_ci(df, dataset, model, layer, title, outpath):
    sub = df[(df.dataset == dataset) & (df.model == model) & (df.layer == layer)]
    if sub.empty:
        return
    agg = (
        sub.groupby("radius")
        .agg(n=("h_norm", "count"), mean=("h_norm", "mean"), sd=("h_norm", "std"))
        .reset_index()
    )
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    agg["lo"] = agg["mean"] - 1.96 * agg["se"]
    agg["hi"] = agg["mean"] + 1.96 * agg["se"]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    rs, mu = agg["radius"].values, agg["mean"].values
    ax.plot(rs, mu, marker="o")
    ax.fill_between(rs, agg["lo"].values, agg["hi"].values, alpha=0.2)
    ax.set_xlabel("loop radius")
    ax.set_ylabel(r"$h_{\mathrm{norm}}=\|H-I\|_F/(2\sqrt{p})$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    ensure_dir(os.path.dirname(outpath))
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# =============================
# RUN BLOCK — toggle as needed
# =============================
ensure_dir(RESULTS_DIR)

RUN_MNIST_MAIN = True
RUN_CIFAR10_MAIN = True
RUN_CIFAR100_MAIN = True
RUN_PLANE_ABL = True
RUN_KQ_SENS = True
RUN_POINTWISE_VS_PATHWISE = True
RUN_ROBUSTNESS = True
RUN_DYNAMICS = True

# ---- 1) MNIST/MLP multi-seed (ERM) ----
if RUN_MNIST_MAIN:
    seeds = [1, 2, 3, 4, 5]
    radii = [0.01, 0.02, 0.05, 0.10, 0.20]
    rows = []
    for s in seeds:
        reg = Regime(epochs=5, batch_size=128, lr=2e-3, weight_decay=1e-4)
        run = train_run(seed=s, dataset="mnist", model_name="mnist_mlp", regime=reg)
        rows += holonomy_curve_for_run_layer(run, "hidden1", radii, k=128, q=64, plane="pca")
        rows += holonomy_curve_for_run_layer(run, "hidden2", radii, k=128, q=64, plane="pca")
        free_run(run)
    mnist_df = pd.DataFrame(rows)
    mnist_df.to_csv(os.path.join(RESULTS_DIR, "mnist_holonomy.csv"), index=False)
    plot_radius_ci(
        mnist_df,
        "mnist",
        "mnist_mlp",
        "hidden1",
        "MNIST/MLP hidden1 — holonomy vs radius",
        os.path.join(RESULTS_DIR, "mnist_h1_ci.png"),
    )
    plot_radius_ci(
        mnist_df,
        "mnist",
        "mnist_mlp",
        "hidden2",
        "MNIST/MLP hidden2 — holonomy vs radius",
        os.path.join(RESULTS_DIR, "mnist_h2_ci.png"),
    )

# ---- 2) CIFAR-10/ResNet-18 multi-seed + regime variants ----
if RUN_CIFAR10_MAIN:
    seeds = [1, 2, 3, 4, 5]
    radii = [0.02, 0.05, 0.10, 0.20]
    regimes = [
        ("ERM", Regime(epochs=8, batch_size=128, lr=1e-3, weight_decay=5e-4, label_smoothing=0.0, mixup_alpha=0.0, adv_train=False)),
        ("LabelSmooth", Regime(epochs=8, batch_size=128, lr=1e-3, weight_decay=5e-4, label_smoothing=0.1)),
        ("Mixup", Regime(epochs=8, batch_size=128, lr=1e-3, weight_decay=5e-4, mixup_alpha=0.2)),
        ("AdvPGD", Regime(epochs=8, batch_size=128, lr=1e-3, weight_decay=5e-4, adv_train=True, pgd_iters=3)),
    ]

    cifar10_csv = os.path.join(RESULTS_DIR, "cifar10_holonomy.csv")
    robust_csv = os.path.join(RESULTS_DIR, "cifar10_robustness.csv")
    have_hol = os.path.exists(cifar10_csv)
    have_rob = os.path.exists(robust_csv)

    def p(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    for name, reg in regimes:
        for s in seeds:
            p(f"=== CIFAR-10 regime={name} seed={s} — training ===")
            run = train_run(seed=s, dataset="cifar10", model_name="resnet18", regime=reg)

            p("Holonomy layer1…")
            rows1 = holonomy_curve_for_run_layer(run, "layer1", radii, k=128, q=64, plane="pca")
            p("Holonomy layer2…")
            rows2 = holonomy_curve_for_run_layer(run, "layer2", radii, k=128, q=64, plane="pca")

            df_rows = pd.DataFrame(rows1 + rows2)
            df_rows["dataset"] = "cifar10"
            df_rows.to_csv(
                cifar10_csv,
                mode=("a" if have_hol else "w"),
                index=False,
                header=not have_hol,
            )
            have_hol = True
            p(f"Appended {len(df_rows)} holonomy rows → {cifar10_csv}")

            p("Robustness eval…")
            clean_acc = accuracy(run.model, run.test_loader)
            fgsm_acc = fgsm_accuracy(run.model, run.test_loader, eps=4 / 255)
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            corr_acc = avg_corruption_accuracy(run.model, run.test_loader.dataset, mean, std)
            pd.DataFrame(
                [
                    dict(
                        regime=name,
                        seed=s,
                        clean_acc=clean_acc,
                        fgsm_acc=fgsm_acc,
                        corr_acc=corr_acc,
                    )
                ]
            ).to_csv(
                robust_csv,
                mode=("a" if have_rob else "w"),
                index=False,
                header=not have_rob,
            )
            have_rob = True
            p(f"Appended robustness row → {robust_csv}")

            free_run(run)

    cifar10_df = pd.read_csv(cifar10_csv)
    plot_radius_ci(
        cifar10_df,
        "cifar10",
        "resnet18",
        "layer1",
        "CIFAR-10/ResNet-18 layer1 — holonomy vs radius",
        os.path.join(RESULTS_DIR, "cifar10_l1_ci.png"),
    )
    plot_radius_ci(
        cifar10_df,
        "cifar10",
        "resnet18",
        "layer2",
        "CIFAR-10/ResNet-18 layer2 — holonomy vs radius",
        os.path.join(RESULTS_DIR, "cifar10_l2_ci.png"),
    )

# ---- 3) CIFAR-100/ResNet-18 ----
if RUN_CIFAR100_MAIN:
    seeds = [1, 2, 3, 4, 5]
    radii = [0.02, 0.05, 0.10, 0.20]
    rows = []
    reg = Regime(epochs=12, batch_size=128, lr=1e-3, weight_decay=5e-4)
    for s in seeds:
        run = train_run(seed=s, dataset="cifar100", model_name="resnet18", regime=reg)
        rows += holonomy_curve_for_run_layer(run, "layer1", radii, k=128, q=64, plane="pca")
        rows += holonomy_curve_for_run_layer(run, "layer2", radii, k=128, q=64, plane="pca")
        free_run(run)
    cifar100_df = pd.DataFrame(rows)
    cifar100_df["dataset"] = "cifar100"
    cifar100_df.to_csv(os.path.join(RESULTS_DIR, "cifar100_holonomy.csv"), index=False)

# ---- 4) Plane ablation on MNIST (single run) ----
if RUN_PLANE_ABL:
    run = train_run(seed=11, dataset="mnist", model_name="mnist_mlp", regime=Regime(epochs=5))
    rows = []
    for pl in ("pca", "random"):
        rows += holonomy_curve_for_run_layer(
            run, "hidden1", [0.02, 0.05, 0.10, 0.20], k=128, q=64, plane=pl
        )
    df_ab = pd.DataFrame(rows)
    df_ab.to_csv(os.path.join(RESULTS_DIR, "mnist_plane_ablation.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for pl, g in df_ab.groupby("plane"):
        g = g.sort_values("radius")
        ax.plot(g["radius"], g["h_norm"], marker="o", label=pl)
    ax.set_xlabel("loop radius")
    ax.set_ylabel(r"$h_{\mathrm{norm}}$")
    ax.set_title("MNIST/MLP hidden1 — PCA-plane vs random-plane")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "mnist_plane_ablation.png"), dpi=180)
    plt.close(fig)
    free_run(run)

# ---- 5) k/q sensitivity ----
if RUN_KQ_SENS:
    run = train_run(seed=12, dataset="mnist", model_name="mnist_mlp", regime=Regime(epochs=5))
    rows = []
    for k in (96, 128, 192):
        for q in (32, 64, 96):
            rows += holonomy_curve_for_run_layer(
                run, "hidden1", [0.10], k=k, q=q, plane="pca"
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "mnist_kq_sensitivity.csv"), index=False
    )
    free_run(run)

# ---- 6) Pointwise (CKA) vs Pathwise (holonomy) ----
if RUN_POINTWISE_VS_PATHWISE:
    A = train_run(seed=21, dataset="mnist", model_name="mnist_mlp", regime=Regime(epochs=5))
    B = train_run(seed=22, dataset="mnist", model_name="mnist_mlp", regime=Regime(epochs=5))
    cka_h1 = cka_after_procrustes(A, B, "hidden1", max_batches=8)
    with open(os.path.join(RESULTS_DIR, "mnist_hidden1_cka.json"), "w") as f:
        json.dump(cka_h1, f, indent=2)
    print("MNIST hidden1 CKA after Procrustes:", cka_h1)
    free_run(A)
    free_run(B)

# ---- 7) Robustness correlations (CIFAR-10 regimes) ----
if RUN_ROBUSTNESS and RUN_CIFAR10_MAIN:
    hol = pd.read_csv(os.path.join(RESULTS_DIR, "cifar10_holonomy.csv"))
    hol_l2_r01 = hol[
        (hol.layer == "layer2") & (np.isclose(hol.radius, 0.10))
    ].copy()
    hol_l2_r01["regime"] = hol_l2_r01["regime"].apply(lambda s: json.loads(s))
    hol_l2_r01["regime_name"] = hol_l2_r01["regime"].apply(
        lambda d: "AdvPGD"
        if d.get("adv_train")
        else (
            "Mixup"
            if d.get("mixup_alpha", 0) > 0
            else ("LabelSmooth" if d.get("label_smoothing", 0) > 0 else "ERM")
        )
    )
    hol_mean = (
        hol_l2_r01.groupby(["regime_name"])
        .h_norm.mean()
        .reset_index()
        .rename(columns={"h_norm": "h_norm_mean"})
    )
    rob = pd.read_csv(os.path.join(RESULTS_DIR, "cifar10_robustness.csv"))
    rob_mean = (
        rob.groupby("regime")
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"regime": "regime_name"})
    )
    merged = hol_mean.merge(rob_mean, on="regime_name", how="inner")
    merged.to_csv(
        os.path.join(RESULTS_DIR, "cifar10_holonomy_vs_robustness.csv"), index=False
    )
    print("Holonomy vs robustness (CIFAR-10):\n", merged)

# ---- 8) Training dynamics: holonomy vs epoch (MNIST hidden1, ERM) ----
if RUN_DYNAMICS:
    base = train_run(seed=33, dataset="mnist", model_name="mnist_mlp", regime=Regime(epochs=1))
    dyn_rows = training_dynamics_curve(base, "hidden1", radii=(0.10,), epochs=5, k=128, q=64)
    pd.DataFrame(dyn_rows).to_csv(
        os.path.join(RESULTS_DIR, "mnist_dynamics.csv"), index=False
    )
    g = pd.DataFrame(dyn_rows).sort_values("epoch")
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.plot(g["epoch"], g["h_norm"], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$h_{\mathrm{norm}}$ @ radius 0.10")
    ax.set_title("MNIST/MLP hidden1 — training dynamics")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "mnist_dynamics.png"), dpi=180)
    plt.close(fig)
    free_run(base)

print("Saved results to:", RESULTS_DIR)
