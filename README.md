# Disease Detection using Deep Learning (Chest CT-Scans)

This is a B.Tech final year project utilizing Deep Learning to detect Chest Diseases (Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal) from CT scans.

## Features
- **Transfer Learning:** Uses `EfficientNetB3` (GPU-optimized) for high-accuracy feature extraction.
- **Advanced Training:** Option to use Stratified K-Fold Cross Validation.
- **Frontend App:** Built with Streamlit, allows users to upload a CT scan for prediction.
- **Explainable AI:** Includes Grad-CAM dynamically applied over the last convolutional layer to highlight the tumor focus area.

---

## 🚀 Important: GPU Setup (NVIDIA RTX)
If the script says "No GPU found by TensorFlow" despite you having an RTX GPU, TensorFlow is likely missing the CUDA libraries in your environment. Follow these steps to fix it on your machine:

1. Ensure you have the newest NVIDIA Game Ready or Studio Drivers installed.
2. If using Windows, install WSL2 (Windows Subsystem for Linux), or install the specific `tensorflow[and-cuda]` package.
   ```bash
   # In your local Python environment:
   pip install "tensorflow[and-cuda]"
   ```
3. Verify GPU is detected by running:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   *If it prints `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`, you are good to go!*

---

## How to Run the Project

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   You must train the model first. It automatically downloads the Kaggle dataset.
   - For fast, standard training: `python train.py`
   - For rigorous, 5-Fold training: `python train_kfold.py`
   *(Both will output a `best_model.h5` file)*

3. **Evaluate the Model (Optional):**
   ```bash
   python evaluate.py
   ```
   This will output the classification metrics (>0.9 expected on GPU) and save `confusion_matrix.png` and `roc_curve.png`.

4. **Launch the Web App:**
   ```bash
   streamlit run app.py
   ```
   Navigate to the local URL (usually `http://localhost:8501`) to interact with the project!
