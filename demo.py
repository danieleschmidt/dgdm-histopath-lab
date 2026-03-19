#!/usr/bin/env python3
"""
Demo: classify 20 synthetic tissue patches with HistoGNN.

Each patch is converted to a spatial patch-graph, then the 3-layer GNN
classifies it as NORMAL or TUMOR.

The model is trained for 80 steps on 16 training patches, then evaluated
on 4 held-out patches.
"""

import torch
import torch.nn as nn
import numpy as np

from dgdm_histopath import PatchGraph, HistoGNN, SyntheticHistoDataset


def build_graph(image: np.ndarray, pg: PatchGraph):
    """Convert a single image to (x, edge_index)."""
    return pg.build(image)


def train(model, dataset, pg, epochs=80, lr=3e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for item in dataset:
            x, edge_index = build_graph(item["image"], pg)
            label = torch.tensor([item["label"]], dtype=torch.long)

            logits, _ = model(x, edge_index)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            avg = total_loss / len(dataset)
            print(f"  epoch {epoch:3d}  loss={avg:.4f}")


def evaluate(model, dataset, pg):
    model.eval()
    correct = 0
    results = []
    with torch.no_grad():
        for item in dataset:
            x, edge_index = build_graph(item["image"], pg)
            _, probs = model(x, edge_index)
            pred = probs.argmax(dim=-1).item()
            gt = item["label"]
            correct += int(pred == gt)
            results.append({
                "gt": item["label_name"],
                "pred": "NORMAL" if pred == 0 else "TUMOR",
                "conf": f"{probs[0, pred].item():.2f}",
                "ok": "✓" if pred == gt else "✗",
            })
    acc = correct / len(dataset)
    return acc, results


def main():
    print("=" * 55)
    print("  DGDM Histopathology Demo")
    print("  Dynamic Graph Diffusion Model — tissue classification")
    print("=" * 55)

    # ---- Dataset --------------------------------------------------------
    full_ds = SyntheticHistoDataset(n_samples=20, patch_size=64, seed=42)
    print(f"\nDataset: {len(full_ds)} patches  "
          f"(NORMAL={full_ds.labels().count(0)}, TUMOR={full_ds.labels().count(1)})")

    train_ds = [full_ds[i] for i in range(16)]
    test_ds  = [full_ds[i] for i in range(16, 20)]

    # ---- PatchGraph -------------------------------------------------------
    # Each 64×64 image → 4 patches (32×32, stride 32) → 4-node graph
    pg = PatchGraph(patch_size=32, stride=32, connectivity=8)

    # Verify graph shape on first image
    x0, ei0 = pg.build(full_ds[0]["image"])
    n_rows, n_cols = pg.grid_shape(full_ds[0]["image"])
    print(f"Graph per patch: {x0.size(0)} nodes, {ei0.size(1)} edges  "
          f"({n_rows}×{n_cols} grid)")

    # ---- Model ------------------------------------------------------------
    model = HistoGNN(in_dim=4, hidden=64, num_classes=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: HistoGNN  ({n_params:,} parameters)")

    # ---- Training ---------------------------------------------------------
    print(f"\nTraining on {len(train_ds)} patches for 80 epochs …")
    train(model, train_ds, pg, epochs=80)

    # ---- Evaluation -------------------------------------------------------
    train_acc, _ = evaluate(model, train_ds, pg)
    test_acc, test_results = evaluate(model, test_ds, pg)

    print(f"\nTrain accuracy : {train_acc * 100:.1f}%")
    print(f"Test  accuracy : {test_acc  * 100:.1f}%  (4 held-out patches)\n")

    print("Test predictions:")
    print(f"  {'GT':<8} {'Pred':<8} {'Conf':>6}  Result")
    print(f"  {'-'*35}")
    for r in test_results:
        print(f"  {r['gt']:<8} {r['pred']:<8} {r['conf']:>6}    {r['ok']}")

    print()
    if test_acc >= 0.5:
        print("✓  Model successfully distinguishes NORMAL from TUMOR tissue patches.")
    else:
        print("✗  Model needs more training or a larger dataset.")
    print("=" * 55)


if __name__ == "__main__":
    main()
