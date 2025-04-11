
# 🔒 Introduction to Gradient-Based Attacks in Text Classification Models

Gradient-based attacks expose inherent vulnerabilities in text classification models, yet existing approaches that rely on discrete text projection often fall short in convergence. In this work, we present a novel framework that leverages continuous optimization in the static embedding space, followed by an embedding projection heuristic—defined by the model's embedding matrix. By combining iterative gradient updates via projected gradient descent (PGD) with a novel embedding projection method, our approach substantially improves adversarial success rates (ASR) and reduces the restriction of iterating only in the textual space. We demonstrate the effectiveness of our method on the toxic classification task using both BERT and ModernBERT architectures, where our attacks significantly reduce classifier confidence on toxic examples.

---

## 🚀 Getting Started

### 🔧 Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset

We use the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip) dataset from Kaggle.

### 📦 Setup
1. Download and unzip the dataset.
2. Place the files inside a `data` folder:
```
data/
├── train.csv
├── test.csv
└── test_labels.csv
```

---

## 🧪 Running Attacks

### ⚡ FGSM Attack
Basic usage:
```bash
python pipeline.py --attack=FGSM --alpha=2 --model_name=bert-base-uncased --test_frac=0.1
```

---

### 🔁 PGD Attack
Standard PGD attack:
```bash
python pipeline.py --attack=PGD --alpha=2 --model_name=bert-base-uncased --test_frac=0.1 --PGD_iterations=50 --return_iter_results=True
```

With embedding projection framework enabled:
```bash
python pipeline.py --attack=PGD --alpha=2 --model_name=bert-base-uncased --test_frac=0.1 --PGD_iterations=50 --return_iter_results=True --enable_emb_proj=True --text_proj_freq=5
```

---

## 📌 Notes
- `--alpha` controls the perturbation step size.
- `--test_frac` sets the fraction of the dataset used for testing.
- `--enable_emb_proj` activates the proposed embedding projection method to enhance robustness.

---

## 📫 Contact
For any questions or feedback, feel free to open an issue or contribute to the project!
