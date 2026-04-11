# Delivery Price Prediction Project

## This project's main goal is to try and predict delivery price, by using specific available data in chosen dataset

Project is for learning purposes of doing proper EDA and use few of the most popular machine learning algorithms. It is fully coded by myself, but some of the functions were inspired by kaggle notebook done by Muh Amri Sidiq.

## Approach

Firstly the most important part was doing Exploratory Data Analysis to understand dataset well. Thanks to that I could recognize
the most important columns and try to do some feature engineering. Since a lot of columns didn't really impact delivery cost,
only few of them were left and only one extra feature was created.

> EDA notebook is available in the [`notebooks/`](notebooks/) directory.


Next step, after preprocessing data was initializing models. In this project I used 3:
* RandomForest
* XGBoost
* LightGBM

All 3 of them are models which are using decision trees, which are perfect at predicting prices mostly using numerical data.

## Results

Three regression models were trained and evaluated using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R² Score**.

### Model Performance

| Model | MAE | MSE | R² |
|---|---|---|---|
| Random Forest | 3.4777 | 22.6007 | 0.9999 |
| XGBoost | 2.6139 | 11.3125 | 0.9999 |
| **LightGBM** | **2.5561** | **10.5902** | **0.9999** |

**LightGBM** achieved the best performance across all metrics and was selected as the final model.

### Feature Importance

All three models consistently identified `delivery_distance_km` as the dominant predictor of delivery price, accounting for over **94–98% of feature importance** in tree-based importance scores. The engineered interaction feature `distance_weight_interaction` ranked second across all models, confirming that combining distance and package weight adds predictive value beyond either feature alone.

Other contributing features included delivery mode (same-day vs. standard vs. two-day), package weight, and package type — while region and vehicle type had minimal impact on price prediction.

> Feature importance plots for all models are available in the [`img/`](img/) directory.

## What I learned

The most important thing for me starting this project was understanding and grasping the idea of doing EDA.
I feel that now it is much easier for me to understand data that is presented to me and I have some basic know-how about to how approach problems. The most
important lesson to me is how EDA is powerful and can have real impact on the project.

## Dataset
Download the Delivery Logistics Dataset (India – Multi-Partner) dataset from Kaggle
https://www.kaggle.com/datasets/kundanbedmutha/delivery-logistics-dataset-india-multi-partner

## How to install and run this project

1. Clone the repo
2. Download dataset and place the files in the `data/` folder
3. Make sure to have installed at least Python 3.10, preferably use conda
4. In your project directory run command ```pip install -r requirements.txt```
5. To run project, all you need is run command ```python main.py```
