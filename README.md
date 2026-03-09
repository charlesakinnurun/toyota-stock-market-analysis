<!--

# Toyota Stock
![Toyota-Stock](/image.jpg)

## Procedures
- Import the libraries
    - pandas
    - numpy
    - seaborn
    - scikit-learn
    - matplotlib
    - yfinance
- Data Acquisition
    - Data acquired from Yahoo Finance API
- Data Loading

| Date       | Price      | Close      | High       | Low        | Open       | Volume |
|------------|------------|------------|------------|------------|------------|--------|
| 2010-01-04 | 57.091179  | 57.151569  | 56.842896  | 56.869738  | 56.869738  | 258800 |
| 2010-01-05 | 56.212124  | 56.299361  | 55.702145  | 55.809507  | 55.809507  | 466000 |
| 2010-01-06 | 56.930134  | 57.084473  | 56.500675  | 56.601331  | 56.601331  | 390000 |
| 2010-01-07 | 56.225525  | 56.259073  | 55.769225  | 55.769225  | 55.769225  | 377700 |
| 2010-01-08 | 57.547462  | 57.634694  | 56.701964  | 56.701964  | 56.701964  | 351900 |
| ...        | ...        | ...        | ...        | ...        | ...        | ...    |
| 2023-12-22 | 170.646301 | 171.324955 | 170.139707 | 170.197055 | 170.197055 | 322800 |
| 2023-12-26 | 171.908020 | 172.405047 | 170.646303 | 170.646303 | 170.646303 | 198500 |
| 2023-12-27 | 172.739609 | 173.112386 | 172.127871 | 172.414626 | 172.414626 | 195500 |
| 2023-12-28 | 173.112381 | 173.666771 | 172.959443 | 173.026359 | 173.026359 | 243000 |
| 2023-12-29 | 175.282150 | 175.760071 | 174.307188 | 175.043190 | 175.043190 | 223200 |



- Feature Engineering
- Pre-Training Visualization

![pre-training-visualization](/pre_training_visualization.png)
- Data Preprocessing
    - Check for missing values
    - Check for duplicated values
    - Drop NaNs values
- Target Definition
- Data Splitting
    - We cannot shuffle the data. We must split it sequentially
    - The model must be trained on the ast train set and tested on the future (test set)
- Data Scaling
    - Using StandardScaler we scale the data to have a mean of 0 and a standard deviation of 1
- Model Comparison
    - Linear Regression
    - Lasso Regression
    - Ridge Regression
    - Support Vector Regression
    - Decision Tree Regressor
    - Random Forest Regressor
- Model Training 
- Hyperparameter Tuning
    - Random Forest
        - n_estimators
        - max_depth
        - min_samples_leaf
    - Suport Vector Regression
        - C
        - gamma
    - Decison Tree Regressor
        - max_depth
        - min_samples_split
        - min_samples_leaf
    - Ridge Regression
        - alpha
    - Lasso Regression
- Post-Training Visualization

![post-training-visualization](/pre_training_visualization.png)
- Prediction Input


## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
    - Anaconda
    - Google Colab
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```



## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/toyota-stock.git
cd toyota-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```


## Project Structure
```
toyota-stock/
│
├── model.ipynb  
|── model.py    
|── blackstone_stock_data.csv  
├── requirements.txt  
├── image.jpg
├── pre_training_visualization.png
├── post_training_visualization.png     
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── SECURITY.md
├── LICENSE
└── README.md          

```

-->

<h1 align="center">Toyota Stock Market Analysis</h1>

![header image](/assets/header.png)