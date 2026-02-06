# NutriClass

> Food name classification based on nutrient values.

Project to classify food names from nutrient features using seven different models. This repository contains the dataset, preprocessing and feature engineering notebooks, trained models, and a prediction script.

**Repository structure**

- Dataset: [Data/synthetic_food_dataset_imbalanced.csv](Data/synthetic_food_dataset_imbalanced.csv)
- Cleaned data: [Data/DataProcessing/cleaned_food_data.csv](Data/DataProcessing/cleaned_food_data.csv)
- Feature engineered data: [Data/FeatureEngineering/feature_engineered_data.csv](Data/FeatureEngineering/feature_engineered_data.csv)
- Notebooks: [NoteBooks](NoteBooks)
- Trained models: [models](models)
- Prediction script: [predict.py](predict.py)

**Models used**

The project trains and evaluates seven classifiers:

- `Support Vector Machine (SVM)`
- `K-Nearest Neighbors (KNN)`
- `Random Forest`
- `XGBoost`
- `Gradient Boosting`
- `Decision Tree`
- `Logistic Regression`

**Best performing model**

- `Support Vector Machine (SVM)` — identified as the best-performing model based on evaluation metrics in the notebooks.

**Quick setup**

1. Create a Python environment (recommended Python 3.8+).
2. Install required packages (example):

```
pip install pandas numpy scikit-learn xgboost jupyter matplotlib seaborn
```

**How to run**

- Explore preprocessing and modeling in the notebooks under [NoteBooks](NoteBooks).
- To run predictions using the script (ensure models exist in `models/`):

```
python predict.py
```

**Notes**

- Trained model files (if saved) are stored in the `models/` directory.
- The notebooks demonstrate data cleaning, feature engineering, training, and evaluation for each model.

If you'd like, I can add a `requirements.txt`, commit the README, or expand sections with example outputs and evaluation metrics.
