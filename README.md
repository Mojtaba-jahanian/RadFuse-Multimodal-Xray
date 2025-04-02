# RadFuse: Multimodal Transformer for Chest X-ray Diagnosis and Retrieval

## 🧠 Introduction
RadFuse is a unified multimodal transformer-based framework designed to jointly model chest radiographs and their associated free-text radiology reports. By combining a Vision Transformer (ViT) for image encoding and ClinicalBERT for text encoding, RadFuse performs:

- Multi-label disease classification
- Image-report retrieval

This repository provides code for training, evaluation, and inference using the RadFuse model, along with visualization utilities for attention heatmaps.

---

## ⚙️ Installation

### 1. Clone the repository:
```bash
git clone https://github.com/YourUsername/RadFuse.git
cd RadFuse
```

### 2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 3. Download datasets:
- **MIMIC-CXR**: [https://physionet.org/content/mimic-cxr/2.0.0/](https://physionet.org/content/mimic-cxr/2.0.0/)
- **IU X-Ray**: [https://openi.nlm.nih.gov/](https://openi.nlm.nih.gov/)

Preprocess the datasets using provided scripts in `data_preprocessing/`.

---

## 🚀 Usage

### Example: Run Inference on an X-ray Image
```bash
python run_inference.py --image_path ./examples/sample_xray.png --report_path ./examples/sample_report.txt
```

### Output:
- Predicted disease labels with confidence scores
- Matched report (retrieval mode)
- Attention heatmap overlay (optional)

---

## 🏋️‍♀️ Train the Model
```bash
python train.py --config configs/train_radfuse.yaml
```

### Configurable Hyperparameters (via YAML):
- `epochs`: number of training epochs
- `batch_size`: size of mini-batches
- `learning_rate`: optimizer learning rate
- `loss_weights`: weights for classification vs retrieval loss

---

## 📊 Evaluate the Model
```bash
python evaluate.py --checkpoint ./checkpoints/model_final.pth
```
Metrics:
- AUC, F1-score for classification
- Recall@10, MRR for retrieval

---

## 📎 Folder Structure
```
RadFuse/
├── configs/              # YAML configs for training and eval
├── data_preprocessing/   # Scripts to preprocess MIMIC-CXR/IU-XRay
├── models/               # Model architecture (ViT, ClinicalBERT, Fusion)
├── training/             # Training, loss functions
├── evaluation/           # Evaluation metrics and loops
├── visualizations/       # Attention visualizer tools
├── examples/             # Sample images and reports
├── checkpoints/          # Saved weights
├── run_inference.py      # Inference script
├── train.py              # Training launcher
├── evaluate.py           # Evaluation launcher
└── requirements.txt      # Python dependencies
```

---

## 👨‍💻 Authors
- Mojtaba Jahanian (Corresponding Author) — Mojtaba160672000@aut.ac.ir
- Abbas Karimi — Abbas.karimi@iau.ac.ir
- Nafiseh Osati Eraghi — Nafiseh.osati@iau.ac.ir
- Faraneh Zarafshan — Fzarafshan@aiau.ac.ir

Affiliations: 
- Department of Computer, Arak Branch, Islamic Azad University, Arak, Iran
- Department of Computer, Ashtian Branch, Islamic Azad University, Arak, Iran

---

## 📄 License
MIT License

---

## ✨ Citation
If you use this code, please cite:
```bibtex
@article{radfuse2025,
  title={Multimodal Transformers for Joint Diagnosis and Retrieval from Chest X-rays and Radiology Reports},
  author={Jahanian, Mojtaba and Karimi, Abbas and Osati Eraghi, Nafiseh and Zarafshan, Faraneh},
  journal={Journal Name},
  year={2025}
}
```

---

## ❓ Questions or Contributions
Feel free to open an issue or submit a pull request. Contributions are welcome!
