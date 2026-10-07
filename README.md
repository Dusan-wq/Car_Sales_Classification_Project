# BMW Car Sales Classification

Machine learning project for predicting BMW car sales performance (High/Low) 
based on attributes such as model, year, region, fuel type, mileage and price.

Developed as part of the **Software Algorithms in Automatic Control Systems (Softverski algoritmi u sistemima automatskog upravljanja)** course.

## Dataset

`BMW_Car_Sales_Classification.csv` — BMW car listings with a binary target 
column `Sales_Classification` (High/Low).

## Documentation

More detailed project report: [`Dokumentacija_za_SAUSAU.pdf`](./Dokumentacija_za_SAUSAU.pdf)

## Technologies

Python · pandas · scikit-learn · XGBoost · seaborn · matplotlib

## Setup & Run

    pip install pandas scikit-learn xgboost seaborn matplotlib
    python Projekat.py

The script performs data cleaning, outlier removal, encoding, class balancing, 
trains and tunes four models (Logistic Regression, Decision Tree, Random Forest, 
XGBoost) with GridSearchCV, and prints accuracy, F1-score and feature importances.
