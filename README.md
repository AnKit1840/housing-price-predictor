# Housing Price Predictor 🏠

## Overview

This project predicts housing prices using machine learning.
The model is trained on housing dataset features such as income, location, and other housing attributes to estimate the median house value.

The project demonstrates the complete machine learning workflow including:

- Data cleaning and preprocessing
- Handling missing values
- Feature engineering
- One-Hot Encoding for categorical features
- Train–test split
- Cross-validation for model validation
- Model training using Scikit-learn
- Model evaluation using RMSE and R² score

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## Project Structure

```
housing-price-predictor
│
├── data/                # Dataset used for training
├── Notebook/            # Jupyter notebooks for analysis and training
├── src/                 # Source code for preprocessing and model training
├── models/              # Saved trained models (ignored in Git)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/AnKit1840/housing-price-predictor.git
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## How to Run

Open the Jupyter notebook and run the training pipeline:

```
jupyter notebook
```

Run the notebook inside the `Notebook` folder to train and evaluate the model.

---
## Models Used

The following machine learning models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation metrics include:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---
## Author

Kit
