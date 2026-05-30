# 🔒 Adversarial Power Sample Generation Tool

This repository provides a tool for generating **adversarial power traces** based on the paper: **_"Post-Silicon Deception: Evasive Hardware Trojan through Adversarial Power Trace"_**.

The goal of this tool is to create adversarial examples capable of deceiving machine learning-based hardware Trojan detectors using power side-channels by manipulating power traces. It supports both **synchronized** and **unsynchronized** adversarial trace generation and includes functionality for **adversarial training** of machine learning models.

---

## 📦 Features

- Generate **synchronized** and **unsynchronized** adversarial power samples  
- Compatible with various hardware Trojan detection benchmarks  
- Supports multiple ML models: HTnet, ResNet, VGG, SVM  
- Optional **adversarial training** for model robustness  
- Customizable noise budget and resolution settings

---

## 🚀 Getting Started

### 1. Install Dependencies

Make sure Python is installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Tool

Use the following command to generate adversarial samples and train the model:

```bash
ipython HW_Trojan_UAP_Generation.py -- \
  --model_name='HTnet' \
  --benchmark='AES-T700' \
  --output_dir='./results/' \
  --sync_epsilon=1.2 \
  --unsync_epsilon=2.4 \
  --resolution=0.1 \
  --sync \
  --unsync \
  --at
```
#### ⚙️ Command-Line Arguments

| Argument            | Description |
|---------------------|-------------|
| `--model_name`      | Choose the model to use. Options: `{HTnet, ResNet-18, SVM, VGG-11}` |
| `--benchmark`       | Select the benchmark. Options: `{AES-T400, AES-T500, AES-T600, AES-T700, AES-T800, AES-T1800, BasicRSA-T200, BasicRSA-T400, PIC16F84-T200}` |
| `--output_dir`      | Directory where generated adversarial samples and trained models will be saved |
| `--sync_epsilon`    | Noise budget for **synchronized** adversarial samples |
| `--unsync_epsilon`  | Noise budget for **unsynchronized** adversarial samples |
| `--resolution`      | Resolution for adversarial sample generation |
| `--sync`            | Enable generation of **synchronized** adversarial samples |
| `--unsync`          | Enable generation of **unsynchronized** adversarial samples |
| `--at`              | Enable **adversarial training** of the selected model |

---

## 📄 Reference

If you use this tool in your research, please cite the following paper by using the following BibTeX entry:

```bibtex
@article{omidi2026post,
  title={Post-Silicon Deception: Evasive Hardware Trojan Through Adversarial Power Trace},
  author={Omidi, Behnam and Alouani, Ihsen and Khasawneh, Khaled N},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2026},
  publisher={IEEE}
}
