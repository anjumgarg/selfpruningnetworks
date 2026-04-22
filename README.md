# selfpruningnetworks
# Self-Pruning Neural Network

This project implements a self-pruning neural network that learns to remove unnecessary weights during training using learnable gates. The model dynamically adapts its structure, improving efficiency while maintaining performance.

---

## 🚀 Overview

Traditional neural networks use all parameters during inference. In this project, each weight is associated with a learnable gate that determines its importance.

- Gate value close to **0** → weight is pruned  
- Gate value close to **1** → weight is retained  

This allows the network to automatically learn a sparse structure.

---

## 🧠 Methodology

### 🔹 Prunable Linear Layer
A custom linear layer is implemented where:

- Each weight has a corresponding gate parameter
- Gates are computed using a sigmoid function
- Effective weights are calculated as:

---

### 🔹 Loss Function

The total loss consists of:

- Classification Loss → CrossEntropyLoss  
- Sparsity Loss → L1 norm of all gate values  

The L1 penalty encourages many gates to become zero, resulting in pruning.

---

## 📊 Results

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 0.0001 | 47.73% | 1.55% |
| 0.001  | 45.85% | 1.70% |
| 0.01   | 41.58% | 1.71% |

> As λ increases, sparsity increases while accuracy decreases, demonstrating a trade-off between model efficiency and performance.

---



## ⚙️ Setup & Installation

Install dependencies:
torch
torchvision
matplotlib
