# -*- coding: utf-8 -*-
"""fproj_2
## **Importing Packages**
"""


# Commented out IPython magic to ensure Python compatibility.

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# Scikit-learn imports.
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Import the main functionality from the SimFin Python API.
import simfin as sf

# Import names used for easy access to SimFin's data-columns.
from simfin.names import *

# ExplainerDashboard Imports
from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *
from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard import ExplainerDashboard

"""# **Loading Data**

Importing data using the **SimFin API** -- a robust stock analytics & data API (simfin.com)
"""

# Setting directory location
sf.set_data_dir('~/simfin_data/')

# Connecting my personal API key
sf.set_api_key(api_key='3a266664-5207-46c9-a10b-38a17ec4f016')

# Selecting the 2 company tickers
tickers = ['LYTS','UIS']

# Add this date-offset to the fundamental data such as
# Income Statements etc., because the REPORT_DATE is not
# when it was actually made available to the public,
# which can be 1, 2 or even 3 months after the Report Date.
offset = pd.DateOffset(days=60)

# Refresh the fundamental datasets (Income Statements etc.)
# every 30 days.
refresh_days = 30

# Refresh the dataset with shareprices every 10 days.
refresh_days_shareprices = 10

# Instantiating hub object to download data
hub = sf.StockHub(tickers=tickers, offset=offset,
                  refresh_days=refresh_days,
                  refresh_days_shareprices=refresh_days_shareprices)

# Loading stock shareprices & Signals
df_prices = hub.load_shareprices(variant='daily')
df_fin_signals = hub.fin_signals(variant='daily')
df_growth_signals = hub.growth_signals(variant='daily')
df_val_signals = hub.val_signals(variant='daily')

# Concating the loaded data to one pandas data frame
dfs = [df_fin_signals, df_growth_signals, df_val_signals, df_prices]
df_main = pd.concat(dfs, axis=1)

"""# **Cleanning the Data**"""

# Data Cleaning Function
def data_clean(df):

  # Remove all rows with only NaN values
  df = df.dropna(how='all').copy()

  # Threshold for the number of rows that must be NaN for each column
  thresh = 0.90 * len(df.dropna(how='all'))

  # Remove all columns which don't have sufficient data.
  df = df.dropna(axis='columns', thresh=thresh)

  # Removing outliers using SimFin Winsorize method
  df = sf.winsorize(df)

  # Dropping Nulls again
  df = df.dropna(how='any')

  return df

# Getting clean data
df_main_clean = data_clean(df_main)

# # Getting respective stock datasets
# lyts_df = df_main_clean.loc['LYTS'].copy()
# uis_df = df_main_clean.loc['UIS'].copy()

"""# **Boiler Code Functions**
---
### **Functions:**

1. **features_added** - Derives all features for our model predictors.

  Function:
  * Creates a column named *'Tomorrow'* by shifting the *'Close'* column back by 1 day
  * Creates a boolean *'Target'* column by determining if *'Tomorrow'* is greater than *'Close'*, thus representing whether the stock increases from tomorrow
  * Calculates rolling means of *'Close'* for 2 days, 3 days, and 30 days

2. **get_feature_names** - Makes a list of features for our model that are filtered by columns that are correlated with the inteded target we are predicting

3. **split_data** - Gets the training and test sets

4. **remove_spaces** - Removes spaces in the feature names, replacing them with "_"
"""

###### FEATURE ENGINEERING #######

def features_added(df):

  # Creating Prediction Target - Will the stock price increase tomorrow?
  df['Tomorrow'] = df['Close'].shift(-1)
  df['Target'] = (df['Tomorrow']>df['Close']).astype(int)


  ###### DERIVING PREDICTOR COLUMNS #######

  # Horizons are the means over the # period (1000 = the last 4 years)
  horizons = [2,3,30]

  # Deriving columns over the horizons list
  for h in horizons:

    # Calculating rolling averages
    rAvg = df.rolling(h).mean()

    # Creating a column that shows how much today's closing price delineates for the past h days average closing price
    rcol = f'How Much Todays Close $ Differs From the past {h}Days Avg Close'
    df[rcol] = df['Close'] / rAvg['Close']

    # Creating column that shows how many times the stock price increased from the last h days
    trend_column = f'How Much The Stock Price Increased From The Last {h}Days'
    df[trend_column] = df.shift(1).rolling(h).sum()['Target']

    # Dropping Nulls from the data
    df.dropna(inplace=True)

  return df

def get_feature_names(df: pd.DataFrame, bottom_limit: float, top_limit:float, target_column: str):
  """
  Get's feature names from data frame
  """
  features = []
  # Iterate through the DataFrame
  for index, row in df.iterrows():
      if bottom_limit <= row[target_column] < top_limit:
          features.append(index)

  if target_column == 'Target':
    remove = ['Open','Low', 'High', 'Close', 'Adj_Close','Tomorrow','Volume']
    features = [x for x in features if x not in remove]
  else:
    remove = ['Open','Low', 'High', 'Close', 'Adj_Close','Target','Volume']
    features = [x for x in features if x not in remove]
  return features

def split_data(df: pd.DataFrame, predictors: list, target_column: str):
  """
  Get's training and test sets
  """
  train = df[-220:].copy()
  test = df[:-220].copy()

  return train[predictors], train[target_column], test[predictors], test[target_column]

def remove_spaces(df):
    df.columns = df.columns.str.replace(' ', '_').str.replace('.','')
    return df

def feature_importance(model, df,features: list,target_column: str):
    """
    Return a DataFrame which compares the signals' Feature
    Importance in the Machine Learning model, to the absolute
    correlation of the signals and stock-returns.

    :param model: Sklearn ensemble model.
    :return: Pandas DataFrame.
    """

        # New column-name for correlation between signals and returns.
    RETURN_CORR = f'{target_column} Correlation'

    # Calculate the correlation between all data-columns.
    df_corr = df.corr()

    # Correlation between signals and returns.
    # Sorted to show the strongest absolute correlations first.
    df_corr_returns = df_corr[target_column] \
                        .abs() \
                        .drop(target_column) \
                        .sort_values(ascending=False) \
                        .rename(RETURN_CORR)

    # Wrap the list of Feature Importance in a Pandas Series.
    df_feat_imp = pd.Series(model.feature_importances_,
                            index=features,
                            name='Feature Importance')

    # Concatenate the DataFrames with Feature Importance
    # and Return Correlation.
    dfs = [df_feat_imp, df_corr_returns]
    df_compare = pd.concat(dfs, axis=1, sort=True)

    # Sort by Feature Importance.
    df_compare.sort_values(by='Feature Importance',
                           ascending=False, inplace=True)

    return df_compare.reset_index()

def plot_feature_importance(model, df,features: list,target_column: str):

  df = feature_importance(model, df,features,target_column)

  # Sort the DataFrame by either "Feature Importance" or "Return Correlation" (you can choose)
  df_sorted = df.sort_values(by="Feature Importance", ascending=False)

  # Select the top 10 features for plotting
  top_features = df_sorted.head(8)

  # Create a bar chart
  plt.figure(figsize=(10, 6))
  plt.barh(top_features["index"], top_features["Feature Importance"], color='b', label='Feature Importance')
  plt.barh(top_features["index"], top_features[f'{target_column} Correlation'], color='r', alpha=0.5, label=f'{target_column} Correlation')
  plt.xlabel('Score')
  plt.title(f'Top 8 Features: Feature Importance vs {target_column} Correlation')
  plt.legend()
  plt.tight_layout()

  plt.show()

def performance(X_test, y_test, y_pred,model, model_type='clf'):

  if model_type == 'clf':
      # Calculate accuracy
      accuracy = accuracy_score(y_test, y_pred)
      print("Accuracy:", accuracy)

      # Calculate confusion matrix
      conf_matrix2 = confusion_matrix(y_test, y_pred)

      # Calculate sensitivity (True Positive Rate)
      sensitivity = conf_matrix2[1, 1] / (conf_matrix2[1, 1] + conf_matrix2[1, 0])
      print("Sensitivity:", sensitivity)

      # Calculate specificity (True Negative Rate)
      specificity = conf_matrix2[0, 0] / (conf_matrix2[0, 0] + conf_matrix2[0, 1])
      print("Specificity:", specificity)

      # Calculate AUC-ROC
      y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
      roc_auc = roc_auc_score(y_test, y_prob)
      print("AUC-ROC:", roc_auc)

      # Calculate ROC curve
      fpr, tpr, thresholds = roc_curve(y_test, y_prob)

      # Plot ROC curve
      plt.figure()
      plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic (ROC)')
      plt.legend(loc="lower right")
      plt.show()

  else:

      # Calculate R-squared
      r2 = r2_score(y_test, y_pred)
      print("R-squared:", r2)

      # Calculate MAE
      mae = mean_absolute_error(y_test, y_pred)
      print("Mean Absolute Error:", mae)

      # Column-name for the models' predicted stock-returns.
      TOTAL_RETURN_PRED = 'Predicted'

      # Create a DataFrame with actual and predicted stock-returns.
      # This is for the training-set.
      df_y_test = pd.DataFrame(y_test)
      df_y_test[TOTAL_RETURN_PRED] = y_pred

      df_y_test.plot()

"""# **LYTS Analysis**

## **Classification**

### **Preparing LYTS data**
"""

# Getting LYTS dataset
lyts_df = df_main_clean.loc['LYTS'].copy()

# Adding Features
lyts_df = features_added(lyts_df)

# Removing spaces from column names
lyts_df = remove_spaces(lyts_df)

# Calcualting column correlations
lyts_corr = lyts_df.corr()

# Creating features list to use as our model predictors for 'Target'
features_lyts_clf = get_feature_names(lyts_corr,
                                      0.005,
                                      0.99,
                                      'Target')

# Getting train & test sets
X_lyts_clf_train, y_lyts_clf_train, X_lyts_clf_test, y_lyts_clf_test = split_data(lyts_df,features_lyts_clf,'Target')

"""### **Creating Model**"""

model_args = \
{
    'n_estimators': 1000,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'n_jobs': -1,
    'random_state': 1234,
}

# Fitting Model
clf_lyts = RandomForestClassifier(**model_args)
clf_lyts.fit(X_lyts_clf_train,y_lyts_clf_train)

"""### **Feature Importance Analysis**"""

plot_feature_importance(clf_lyts,
                        lyts_df,
                        features_lyts_clf,
                        'Target')

"""### **Model Performance Analysis**"""

y_lyts_pred = clf_lyts.predict(X_lyts_clf_test)

performance(X_lyts_clf_test,y_lyts_clf_test,y_lyts_pred,clf_lyts)

"""# **UIS Analysis**

## **Classification**

### **Preparing UIS data**
"""

# Getting LYTS dataset
uis_df = df_main_clean.loc['UIS'].copy()

# Adding Features
uis_df = features_added(uis_df)

# Removing spaces from column names
uis_df = remove_spaces(uis_df)

# Calcualting column correlations
uis_corr = uis_df.corr()

# Creating features list to use as our model predictors for 'Target'
features_uis_clf = get_feature_names(uis_corr,
                                      0.005,
                                      0.99,
                                      'Target')

# Getting train & test sets
X_uis_clf_train, y_uis_clf_train, X_uis_clf_test, y_uis_clf_test = split_data(uis_df,features_uis_clf,'Target')

X_uis_clf_test.columns.shape

"""### **Creating Model**"""

model_args = \
{
    'n_estimators': 1000,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'n_jobs': -1,
    'random_state': 1234,
}

# Fitting Model
clf_uis = RandomForestClassifier(**model_args)
clf_uis.fit(X_uis_clf_train,y_uis_clf_train)

"""### **Feature Importance Analysis**"""

plot_feature_importance(clf_uis,
                        uis_df,
                        features_uis_clf,
                        'Target')

"""### **Model Performance**"""

y_uis_clf_pred = clf_uis.predict(X_uis_clf_test)

performance(X_uis_clf_test,y_uis_clf_test,y_uis_clf_pred,clf_uis,model_type='clf')

"""## **Regression**"""

# Creating features list to use as our model predictors for 'Target'
features_uis_regr = get_feature_names(uis_corr,
                                      0.005,
                                      0.99,
                                      'Tomorrow')

# Getting train & test sets
X_uis_regr_train, y_uis_regr_train, X_uis_regr_test, y_uis_regr_test = split_data(uis_df,features_uis_regr,'Tomorrow')

model_args = \
{
    'n_estimators': 1000,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'n_jobs': -1,
    'random_state': 1234,
}

# Fitting Model
regr_uis = RandomForestRegressor(**model_args)
regr_uis.fit(X_uis_regr_train,y_uis_regr_train)

plot_feature_importance(regr_uis,
                        uis_df,
                        features_uis_regr,
                        'Tomorrow')

y_uis_regr_pred = regr_uis.predict(X_uis_regr_test)

performance(X_uis_regr_test,y_uis_regr_test,y_uis_regr_pred,regr_uis,model_type='regr')

"""# **Dashboards**"""


explainer = ClassifierExplainer(clf_lyts, X_lyts_clf_test, y_lyts_clf_test)
db = ExplainerDashboard(explainer, title="Cool Title")
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)

db = ExplainerDashboard.from_config("dashboard.yaml")
app = db.flask_server()
