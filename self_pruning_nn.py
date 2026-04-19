"""
Self-Pruning Neural Network — Tredence AI Engineering Internship Case Study
============================================================================
Author  : Vaikunth
Dataset : CIFAR-10
Task    : Image Classification with learnable weight pruning via sigmoid gates

Overview
--------
Each weight in every linear layer is paired with a learnable "gate score".
A sigmoid transforms the gate score into a gate ∈ (0, 1).
The effective weight = weight * gate.
An L1 sparsity penalty on all gates drives many of them toward 0, effectively
removing (pruning) the corresponding weights during training itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

import time
import math


# ──────────────────────────────────────────────────────────────────────────────
# PART 1 — PrunableLinear layer
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies every weight by a
    learnable gate produced by sigmoid(gate_score).

    During forward:
        gates        = sigmoid(gate_scores)          # shape: (out, in)
        pruned_w     = weight * gates                # element-wise
        output       = x @ pruned_w.T + bias

    Gradients flow through both `weight` and `gate_scores` automatically
    because all operations are differentiable w.r.t. both tensors.

    When a gate collapses to ≈ 0 the corresponding weight contributes
    nothing to the output — the connection is effectively pruned.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias — same initialisation as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Gate scores — one scalar per weight; initialised near 0 so gates
        # start close to sigmoid(0) = 0.5 (all connections half-open at init)
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        # Kaiming uniform for weights (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Small positive init so gates start slightly above 0.5
        nn.init.constant_(self.gate_scores, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: turn gate_scores into values in (0, 1)
        gates = torch.sigmoid(self.gate_scores)          # (out, in)

        # Step 2: gate each weight element-wise
        pruned_weights = self.weight * gates             # (out, in)

        # Step 3: standard linear operation — gradients flow through both
        #         `self.weight` and `self.gate_scores` via `pruned_weights`
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from the graph)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_fraction(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (i.e. effectively pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ──────────────────────────────────────────────────────────────────────────────
# Network definition
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32 RGB → 10 classes).
    Every Linear layer is replaced by PrunableLinear so the entire
    network can learn to prune its own weights.

    Architecture (chosen to be wide enough that pruning is meaningful):
        Flatten → 3072
        PrunableLinear 3072 → 1024  + BatchNorm + ReLU + Dropout
        PrunableLinear 1024 → 512   + BatchNorm + ReLU + Dropout
        PrunableLinear 512  → 256   + BatchNorm + ReLU
        PrunableLinear 256  → 10    (logits)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def prunable_layers(self):
        """Yield every PrunableLinear module in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across ALL PrunableLinear layers.

        Why L1?  The L1 norm sums |gate_i| for every gate.  Because gates are
        already in (0,1) via sigmoid, |gate_i| = gate_i, so this is the sum of
        all gate values.  The gradient of |gate_i| w.r.t. gate_i is +1 wherever
        gate_i > 0 — a constant push toward zero, unlike the L2 gradient which
        shrinks as the value shrinks and therefore never reaches exactly 0.
        The L1 penalty is the canonical sparsity inducer.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)   # keep in computation graph
            total = total + gates.abs().sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Percentage of gates (across all layers) below `threshold`."""
        all_gates = torch.cat([
            layer.get_gates().view(-1) for layer in self.prunable_layers()
        ])
        return (all_gates < threshold).float().mean().item() * 100.0

    def all_gate_values(self) -> np.ndarray:
        """Collect every gate value into a single NumPy array."""
        return torch.cat([
            layer.get_gates().view(-1) for layer in self.prunable_layers()
        ]).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# PART 2 — Data loading
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 256):
    """Download CIFAR-10 and return (train_loader, test_loader)."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# PART 3 — Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, lam: float):
    """
    Run one full epoch.

    Total Loss = CrossEntropyLoss(logits, labels)
               + λ × L1_norm_of_all_gates
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)                              # forward pass
        clf_loss     = F.cross_entropy(logits, labels)     # classification term
        sparse_loss  = model.sparsity_loss()               # sparsity term
        loss         = clf_loss + lam * sparse_loss        # combined objective

        loss.backward()                                     # back-prop through gates
        optimizer.step()

        batch_size   = labels.size(0)
        total_loss  += loss.item() * batch_size
        correct     += logits.argmax(dim=1).eq(labels).sum().item()
        n_samples   += batch_size

    avg_loss = total_loss / n_samples
    accuracy = correct   / n_samples * 100.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute test accuracy."""
    model.eval()
    correct   = 0
    n_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds      = model(images).argmax(dim=1)
        correct   += preds.eq(labels).sum().item()
        n_samples += labels.size(0)
    return correct / n_samples * 100.0


def train_and_evaluate(lam: float,
                       epochs: int,
                       train_loader,
                       test_loader,
                       device,
                       lr: float = 3e-3) -> dict:
    """
    Train a fresh SelfPruningNet with a given λ and return results.

    Returns
    -------
    dict with keys: lam, test_acc, sparsity, gate_values, history
    """
    print(f"\n{'='*60}")
    print(f"  λ = {lam}   |   epochs = {epochs}   |   lr = {lr}")
    print(f"{'='*60}")

    model = SelfPruningNet(dropout=0.3).to(device)

    # Adam updates both weights and gate_scores simultaneously
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing smoothly decays lr to near-zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "sparsity": []}
    t0      = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, lam)
        te_acc          = evaluate(model, test_loader, device)
        sparsity        = model.overall_sparsity()
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["sparsity"].append(sparsity)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={tr_loss:.4f}  train={tr_acc:.1f}%  "
                  f"test={te_acc:.1f}%  sparsity={sparsity:.1f}%  "
                  f"[{elapsed:.0f}s]")

    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity()
    gate_values    = model.all_gate_values()

    print(f"\n  ✔  Final test accuracy : {final_test_acc:.2f}%")
    print(f"  ✔  Final sparsity      : {final_sparsity:.2f}%  "
          f"(gates < 1e-2)")

    return {
        "lam":         lam,
        "test_acc":    final_test_acc,
        "sparsity":    final_sparsity,
        "gate_values": gate_values,
        "history":     history,
        "model":       model,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 4 — Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(results: list, save_path: str = "gate_distributions.png"):
    """
    For each experiment, plot a histogram of final gate values.
    A successful run shows a large spike near 0 (pruned weights) and a
    secondary cluster away from 0 (surviving weights).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, res, color in zip(axes, results, colors):
        gates = res["gate_values"]
        ax.hist(gates, bins=80, color=color, edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_title(
            f"λ = {res['lam']}\n"
            f"Test acc = {res['test_acc']:.1f}%  |  Sparsity = {res['sparsity']:.1f}%",
            fontsize=10,
        )
        ax.set_xlabel("Gate value")
        ax.set_ylabel("Count")
        ax.axvline(x=1e-2, color="red", linestyle="--", linewidth=1.2,
                   label="Prune threshold (1e-2)")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of Final Gate Values\n(spike at 0 ↔ pruned connections)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Plot saved → {save_path}]")


def plot_training_curves(results: list, save_path: str = "training_curves.png"):
    """Plot test accuracy and sparsity over epochs for all λ values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    styles = ["-", "--", "-."]

    for res, color, style in zip(results, colors, styles):
        lam  = res["lam"]
        hist = res["history"]
        eps  = range(1, len(hist["test_acc"]) + 1)

        ax1.plot(eps, hist["test_acc"],  color=color, linestyle=style, label=f"λ={lam}")
        ax2.plot(eps, hist["sparsity"],  color=color, linestyle=style, label=f"λ={lam}")

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test Accuracy (%)"); ax1.set_title("Test Accuracy vs Epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Sparsity (%)")     ; ax2.set_title("Sparsity vs Epoch")
    ax1.legend(); ax2.legend()
    ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)

    fig.suptitle("Training Curves — Self-Pruning Network", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot saved → {save_path}]")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    LAMBDAS    = [1e-5, 1e-4, 1e-3]   # low / medium / high sparsity pressure
    EPOCHS     = 40
    BATCH_SIZE = 256
    LR         = 3e-3

    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n  Device : {device}")
    torch.manual_seed(42)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # ── Experiments ───────────────────────────────────────────────────────────
    all_results = []
    for lam in LAMBDAS:
        res = train_and_evaluate(
            lam=lam,
            epochs=EPOCHS,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=LR,
        )
        all_results.append(res)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print(f"  {'Lambda':<12}{'Test Accuracy (%)':>20}{'Sparsity Level (%)':>22}")
    print("="*55)
    for res in all_results:
        print(f"  {res['lam']:<12.0e}{res['test_acc']:>20.2f}{res['sparsity']:>22.2f}")
    print("="*55)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_gate_distribution(all_results, "gate_distributions.png")
    plot_training_curves(all_results,   "training_curves.png")

    print("\n  All done!  Check gate_distributions.png and training_curves.png\n")


if __name__ == "__main__":
    main()
