# House Price Predictor - CMPT 310

## Environment Setup
To run this project, we recommend using a virtual environment. Install the required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```
Running the Machine Learning Pipeline
Our project includes scripts for both unsupervised clustering and supervised regression testing.

1. Baseline Model Evaluation
To run the dummy regressor (mean strategy) and output the baseline Train and Validation MAPE:

```bash
python baseline.py
```

2. K-Means Clustering
To map the dataset boundaries and generate the optimal $k=2$ cluster profiles:

```bash
python K_Means_Clustering.py
```

3. Linear & Non-Linear Regression
To execute our supervised learning models (Linear, Polynomial, Piecewise) across the clusters and output the evaluation metrics (MAPE, RMSE, MAE, R²):

```bash
python linear_regression.py
```

4. Decision Tree Regression
To execute the global tree-based model, find the optimal hyperparameters, and evaluate its performance:

```bash
python treeregression.py
```

Running the Web Application (Live Demo)
We have built a Flask web interface that actively trains our Best-In-Cluster prediction pipeline on startup. When a user inputs property details, the app dynamically determines the house's K-Means cluster, k=2, and routes it to the correct champion model (Polynomial Degree 2 for Cluster 0, or Decision Tree for Cluster 1) for the final price prediction.
1. Navigate to the interface folder and start the Flask server:

```bash
cd interface
python app.py
```

Open your web browser and navigate to http://127.0.0.1:5000/.

Select your property features from the dropdowns and click "Predict Market Price".