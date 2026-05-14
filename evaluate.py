import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

def evaluate():
    model_path = 'best_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, compile=False)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    if not os.path.exists("test_data.csv"):
        print("test_data.csv not found. Please run train.py first to generate the test split.")
        return

    test_df = pd.read_csv("test_data.csv")

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("Evaluating model on test dataset...")
    results = model.evaluate(test_generator, verbose=1)

    loss = results[0]
    accuracy = results[1]

    if len(results) > 2:
        precision = results[2]
        recall = results[3]
    else:
        Y_pred = model.predict(test_generator, verbose=0)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = test_generator.classes
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    print("\n--- Evaluation Metrics ---")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    test_generator.reset()
    print("\nGenerating predictions for Confusion Matrix and ROC curve...")
    Y_pred = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)

    class_indices = test_generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    y_true = test_generator.classes
    class_labels = [index_to_class[i] for i in range(len(class_indices))]

    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))

    print("Generating ROC Curve...")
    n_classes = len(class_labels)
    Y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        if np.sum(Y_true_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(Y_true_bin[:, i], Y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], roc_auc[i] = [0], [0], 0

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    if n_classes == 2:
        colors = ['blue', 'red']

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    print("Saved roc_curve.png")

if __name__ == '__main__':
    evaluate()
