# COVID-19 & Pneumonia Detection using Deep Learning (Chest X-Rays)

This is a B.Tech final year project utilizing Deep Learning to perform multi-class classification of Chest Diseases (COVID, Lung Opacity, Viral Pneumonia, Normal) from X-ray scans.

## Features
- **Massive Dataset:** Uses the renowned `COVID-19 Radiography Database` (over 21,000 images) to train a highly robust model, avoiding class-imbalance and small-sample overfitting entirely.
- **Transfer Learning:** Uses `ResNet50V2` (GPU-optimized) for high-accuracy feature extraction.
- **Advanced Training:** Option to use Stratified K-Fold Cross Validation.
- **Class Weights:** Automatically balances penalties for any residual underrepresented classes to prevent prediction bias.
- **Frontend App:** Built with Streamlit, allows users to upload a chest X-ray scan for prediction.
- **Explainable AI:** Includes Grad-CAM dynamically applied over the last convolutional layer to highlight the opaque/white areas in the lungs indicating infection.

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

3. **Evaluate the Model:**
   ```bash
   python evaluate.py
   ```
   This will test your `best_model.h5` against the hold-out `test_data.csv` generated during training. It outputs classification metrics (>0.9 expected) and saves `confusion_matrix.png` and `roc_curve.png`.

4. **Launch the Web App:**
   ```bash
   streamlit run app.py
   ```
   Navigate to the local URL (usually `http://localhost:8501`) to interact with the project!
