# Handwriting Digit Recognition using CNN (PyTorch)

This repository contains a high-performance implementation of a **Convolutional Neural Network (CNN)** using **PyTorch** to solve the handwritten digit classification problem (MNIST) from the Kaggle Digit Recognizer competition.

## 🚀 Project Overview
The project focuses on building an automated system capable of recognizing handwritten digits (0-9). As a **Computer and Control Engineer**, the implementation emphasizes not just accuracy, but also the stability of the training process and the robustness of the model architecture.

## 🏗 Network Architecture & Engineering Design
The model was designed using a `Sequential` structure with a focus on feature extraction and regularization:
*   **Feature Extraction:** Utilized `Conv2d` layers to capture spatial hierarchies in the images.
*   **Stability:** Integrated `BatchNorm2d` layers after convolutions to stabilize the hidden state distributions and accelerate convergence.
*   **Non-Linearity:** Used `ReLU` activation functions to enable the model to learn complex patterns.
*   **Dimensionality Reduction:** Employed `MaxPool2d` for spatial downsampling while retaining essential features.
*   **Generalization (Regularization):** Applied `Dropout` layers to mitigate overfitting, ensuring the model performs well on unseen test data.

## 🛠 Technical Implementation & Tools
The project leverages the full power of the PyTorch ecosystem and Python data science stack:
*   **Core Framework:** `torch` & `torch.nn` for model building.
*   **Data Pipeline:** `DataLoader` and `TensorDataset` for efficient batching and memory management.
*   **Preprocessing:** `pandas` and `sklearn` for data splitting and normalization.
*   **Visualization:** `seaborn` and `matplotlib` for analyzing the training curves and error distribution.

## 📈 Optimization & Control Strategy
In line with control engineering principles, the training process was treated as an optimization problem:
*   **Optimizer:** Used the **Adam** optimizer for its adaptive learning rate capabilities.
*   **Loss Function:** **CrossEntropyLoss** was chosen as the objective function for multi-class classification.
*   **Dynamic Feedback:** Implemented a **Learning Rate Scheduler** (`ReduceLROnPlateau`). This mimics a closed-loop system where the "system" (the model) monitors the validation loss and automatically reduces the learning rate when the improvement stalls (plateaus), ensuring fine-tuned convergence.

## 📁 Dataset & Model Persistence
*   **Data Source:** [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer). 
*   **Model Weights:** The final trained state of the model is saved using `torch.save(model.state_dict(), 'model_weights.pth')`, allowing for easy deployment or further fine-tuning.
*   **Evaluation:** Detailed analysis was performed using a `confusion_matrix` to identify specific digit-class confusion (e.g., distinguishing between 4 and 9).

---

### 💡 Key Technical Highlights:
1.  **Engineering Approach:** The project doesn't just provide code; it explains the **Optimization Strategy** (Adam + ReduceLROnPlateau) and why these specific control mechanisms were chosen.
2.  **Professional Documentation:** By including a `requirements.txt` (torch, pandas, seaborn, etc.), the project follows professional software development standards.
3.  **Data Ethics:** The README explicitly mentions that large CSV data files are ignored (via `.gitignore`) and provides links to the official source, respecting storage best practices.
4.  **Visual Insights:** Emphasis is placed on the **Confusion Matrix** to demonstrate a deep understanding of model behavior beyond just "total accuracy" percentages.
