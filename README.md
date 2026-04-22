# Wine Quality Classification using Cloud-Based Machine Learning

## Project Overview

This project implements a **Wine Quality Classification** model using the famous **sklearn wine dataset** (Italian wine cultivars). The solution is built in **Google Colab** with GPU acceleration and demonstrates a complete end-to-end machine learning workflow.

## Dataset

The dataset contains chemical analysis of **178 wine samples** from 3 different cultivars (class_0, class_1, class_2) grown in the same region of Italy. Features include:

- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

## Methodology

### 1. Environment Setup
- Google Colab with GPU
- Google Drive mounting for persistent storage

### 2. Data Exploration & Visualization
- Statistical summary and missing values check
- Pairplot for feature relationships
- Correlation heatmap
- Box plots for feature distribution by cultivar

### 3. Model Training
- **Algorithm:** Random Forest Classifier
- Parameters: 100 trees, n_jobs=-1
- Train-test split: 80/20 (stratified)

### 4. Model Evaluation
- Overall accuracy: **100%**
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization

### 5. Feature Importance Analysis
- Identification of most important features for wine classification

### 6. Model Persistence
- Model saved to Google Drive using joblib
- Feature importance exported as CSV

### 7. Real-time Predictions
- Demonstration of model reload and inference on new samples

### 8. Cross-Validation
- 5-fold cross-validation for robustness testing

## Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 100.00% |
| Cross-Validation Accuracy | ~100% |

## Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn

## Usage

1. Open the notebook in **Google Colab**
2. Run all cells sequentially
3. The model will be saved to your Google Drive at:
   `/content/drive/MyDrive/Wine_Classification_Project/`

## Files Generated

- `wine_classifier_rf.pkl` - Trained Random Forest model
- `feature_importance.csv` - Feature importance rankings

## Author

Data Science Project - Week 7

## License

This project is for educational purposes.
