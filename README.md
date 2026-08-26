# 🕵️ AI Deepfake & Synthetic Image Detector

A deep learning model that classifies face photos as **REAL** or **FAKE (AI-generated/deepfake)**, built with transfer learning on EfficientNet, combined with Grad-CAM explainability and a live interactive demo.

**🔗 Live Demo:** [Add your Hugging Face Space / Gradio link here]
**📓 Notebook:** `deepfake_detector_project.ipynb`

---

## 📌 Overview

Deepfakes and AI-generated synthetic images are increasingly used for scams, misinformation, identity fraud, and non-consensual content. This project builds a practical, end-to-end deep learning pipeline to detect such content, inspired by recent research in unified face-forgery detection (see [Base Paper](#base-paper--inspiration) below).

The project was built as a learning-focused deep learning project, prioritizing understanding every step of the pipeline (data handling, transfer learning, training, evaluation, explainability, deployment) over replicating a complex research architecture.

## 🧠 How It Works

1. A face/image is uploaded
2. The image is resized and normalized
3. A pretrained **EfficientNet-B0** CNN (fine-tuned via transfer learning) extracts visual features
4. A classification head outputs **REAL** or **FAKE** with a confidence score
5. **Grad-CAM** visualizes which regions of the image most influenced the prediction

## 📊 Datasets Used

| Dataset | Size | Source | Purpose |
|---|---|---|---|
| 140k Real and Fake Faces | 100,000 train / 20,000 valid images | [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | Face-swap style deepfake detection (GAN-generated faces) |
| AI vs. Human-Generated Images | 79,950 images | [Kaggle](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset) | Broader AI-generated image detection (modern generative models) |

Both datasets were combined (**179,950 total training images**) to improve generalization beyond a single deepfake generation technique.

## 🏗️ Model Architecture

- **Base model:** EfficientNet-B0 (pretrained on ImageNet)
- **Approach:** Transfer learning — final classification layer replaced and fine-tuned for binary classification (real vs. fake)
- **Input size:** 128×128 RGB images
- **Loss function:** Cross-Entropy Loss
- **Optimizer:** Adam (learning rate = 0.0001)
- **Training:** 3 epochs, batch size 32

## 📈 Results

| Model Version | Training Data | Train Accuracy | Validation Accuracy |
|---|---|---|---|
| v1 | 140k Faces only (100k images) | 98.98% | 98.83% |
| **v2 (final)** | **Combined (179,950 images)** | **99.56%** | **99.22%** |

Combining the two datasets improved validation accuracy without any loss in performance on the original face-deepfake task (no catastrophic forgetting observed).

## 🔍 Explainability (Grad-CAM)

Grad-CAM was used to visualize which image regions the model focuses on when making predictions. This surfaced a genuine limitation, discussed below.

## ⚠️ Honest Limitations

Real-world testing (photos outside the training distribution) revealed important limitations, which are documented here rather than hidden:

- **Overconfidence on out-of-distribution images:** The model sometimes outputs very high confidence (near 100%) on images stylistically different from training data (e.g., professionally-lit social media / promotional photography), which is a known miscalibration issue in deepfake detectors generalizing to unseen generation methods and photography styles.
- **Attention not always face-restricted:** Grad-CAM visualizations showed the model sometimes attends to clothing/background regions rather than exclusively facial features — suggesting a face-cropping preprocessing step (as used in professional pipelines, e.g., YOLOv5-based face detection) would likely improve reliability.
- **Reduced accuracy on modern/professional photography styles:** Performance on hand-picked real-world test photos (~60–80% correct) was noticeably lower than the formal validation accuracy (99.22%), highlighting the generalization gap between benchmark datasets and real-world deployment conditions.

These findings are consistent with active, unsolved challenges in deepfake detection research and are documented here transparently as a basis for future improvement rather than treated as a final result.

## 🚀 Future Work

- Add a face-cropping preprocessing step (e.g., YOLOv5/MTCNN) before classification, to restrict model attention to facial regions
- Explore frequency-domain / texture-based artifact detection (e.g., Error Level Analysis, FFT-based features) to complement CNN-based spatial features
- Confidence calibration techniques to reduce overconfident incorrect predictions
- Extend from single images to video (frame-by-frame + temporal modeling, e.g., BLSTM) for deepfake video detection
- Incorporate more diverse, modern AI-generator sources (diffusion models, newer GANs) into training data

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Data Handling:** Kaggle API, Pandas, PIL
- **Explainability:** pytorch-grad-cam
- **Demo/Deployment:** Gradio, Hugging Face Spaces
- **Environment:** Google Colab (T4 GPU), Google Drive (model persistence)

## 📂 Repository Structure

```
├── deepfake_detector_project.ipynb   # Full training & evaluation notebook
├── deepfake_model_v2.pth             # Final trained model weights
├── README.md                         # This file
```

## ▶️ How to Run

1. Open `deepfake_detector_project.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells (requires a Kaggle API key — `kaggle.json`)
4. The final cells launch a Gradio demo with a live shareable link

## 📖 Base Paper / Inspiration

This project was initially inspired by:
> Sekar & Rajkumar, "An optimal deep learning-based face forgery detection system using DCXCN-BLSTM," *Soft Computing*, 2026. [DOI: 10.1007/s00500-026-11218-w](https://doi.org/10.1007/s00500-026-11218-w)

The referenced paper proposes a more complex unified detection system (YOLOv5 preprocessing + Gabor/LBP texture features + deformable-convolution Xception + BLSTM + Chicken Swarm Optimization) covering digital, adversarial, and physical spoofing attacks. This project implements a scoped-down, beginner-appropriate version focused on digital image classification, with the original paper's techniques noted as future work directions above.

## 👤 Author

NAME-Ankit kumar 

## Connect with me 
linkedIn-https://www.linkedin.com/in/ankit-kumar-490051397/

---

*This project was built as a hands-on deep learning learning exercise. Contributions, feedback, and suggestions are welcome.*
