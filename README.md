# Linear Regression

## 📌 Objective
To implement and understand simple & multiple linear regression using a housing price prediction dataset.

## 🛠 Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 📊 Dataset
Used: [Housing Price Prediction Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)  
File: `Housing.csv`

## 📈 Steps Performed
1. Loaded and preprocessed the dataset (handled missing values and categorical features using one-hot encoding).
2. Split the data into training and testing sets.
3. Trained a multiple linear regression model using `LinearRegression` from `sklearn`.
4. Evaluated the model using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score
5. Printed model coefficients for interpretation.
6. Visualized model performance using an **Actual vs Predicted** scatter plot saved as `regression_plot.png`.

## 📷 Output Visualization
The plot (`regression_plot.png`) compares predicted prices with actual prices. Points close to the red dashed line indicate better prediction accuracy.

## ▶️ How to Run
1. Download the dataset and save `Housing.csv` in the same directory as the script.
2. Run the script using:

```bash
python main.py
