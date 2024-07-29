
##############################
# Telco Customer Churn Feature Engineering
##############################

# Problem: A machine learning model is to be developed to predict customers who will leave the company.
# Before developing the model, you are expected to perform the necessary data analysis and feature engineering steps.

# Telco customer churn data contains information about a fictional telecommunications company providing home phone and Internet services
# to 7043 customers in California in the third quarter. It includes information on which customers have left, stayed, or signed up for services.

# 21 Variables 7043 Observations

##################################
# Required Libraries and Functions
##################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import optuna
from sklearn.metrics import precision_score, f1_score, accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import random
import warnings

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

##################################
# 1. EXPLOTARY DATA ANALYSIS
##################################

##################################
# OVERVİEW
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



##################################
# CAPTURING NUMERICAL AND CATEGORICAL VARIABLES
##################################

df.dtypes
df.customerID.nunique() # 7043

categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_but_categorical = [x for x in df.columns if df[x].nunique() < 5 and df[x].dtypes != "O"]
categorical_features = categorical_features + numerical_but_categorical

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features = [x for x in numerical_features if x not in numerical_but_categorical]

categorical_features
numerical_features


##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in categorical_features[1:]:
    cat_summary(df, col)



##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in numerical_features:
    num_summary(df, col, plot=True)


df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show()


df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two year")
plt.show()


##################################
# ANALYSIS OF NUMERICAL VARIABLES ACCORDING TO TARGET
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in numerical_features:
    target_summary_with_num(df, "Churn", col)



##################################
# ANALYSIS OF CATEGORICAL VARIABLES ACCORDING TO TARGET
##################################


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in categorical_features[1:]:
    target_summary_with_cat(df, "Churn", col)




##################################
# CORRELATION
##################################

df[numerical_features].corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[numerical_features].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# It is observed that TotalCharges is highly correlated with MonthlyCharges and tenure.

df[numerical_features].corrwith(df["Churn"]).sort_values(ascending=False)

##################################
# 2. FEATURE ENGINEERING
##################################

##################################
# MISSING VALUE ANALYSIS
##################################

df.isnull().sum()

df[["tenure", "MonthlyCharges", "TotalCharges"]].head()

df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"], inplace=True)

df.isnull().sum()



##################################
# BASE MODEL
##################################

dff = df.copy()
categorical_features = [x for x in categorical_features if x not in ["Churn", "customerID"]]
categorical_features

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, categorical_features, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345, verbosity=-1)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# ########## LR ##########
# Accuracy: 0.8021
# Auc: 0.8423
# Recall: 0.5324
# Precision: 0.657
# F1: 0.588
# ########## KNN ##########
# Accuracy: 0.7637
# Auc: 0.7467
# Recall: 0.4462
# Precision: 0.5711
# F1: 0.5005
# ########## CART ##########
# Accuracy: 0.7263
# Auc: 0.6576
# Recall: 0.5067
# Precision: 0.4849
# F1: 0.4952
# ########## RF ##########
# Accuracy: 0.7928
# Auc: 0.8253
# Recall: 0.4869
# Precision: 0.6463
# F1: 0.5553
# ########## SVM ##########
# Accuracy: 0.7694
# Auc: 0.7141
# Recall: 0.2905
# Precision: 0.6488
# F1: 0.4007
# ########## XGB ##########
# Accuracy: 0.7857
# Auc: 0.8238
# Recall: 0.527
# Precision: 0.6133
# F1: 0.5665
# ########## LightGBM ##########
# Accuracy: 0.7968
# Auc: 0.8365
# Recall: 0.5319
# Precision: 0.6426
# F1: 0.5817
# ########## CatBoost ##########
# Accuracy: 0.7985
# Auc: 0.8406
# Recall: 0.5078
# Precision: 0.6566
# F1: 0.5723


##################################
# FEATURE EXTRACTION
##################################

# To create an annual categorical variable from the tenure variable
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"


# Labeling customers with 1 or 2-year contracts as Engaged
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# People who do not receive any support, backup, or protection
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Customers with a monthly contract who are young
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# The total number of services received by the person
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)


# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# People who receive any streaming service
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Increase of the current price compared to the average price
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Fee per service
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)


df.head()
df.shape


##################################
# ENCODING
##################################

# Separating variables based on their types
cat_features = df.select_dtypes(include=['object']).columns.tolist()
num_but_cat = [x for x in df.columns if df[x].nunique() < 5 and df[x].dtypes != "O"]
cat_features = cat_features + num_but_cat
cat_features

# Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [x for x in df.columns if df[x].dtypes == "O" and df[x].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding
cat_features = [x for x in cat_features if x not in binary_cols and x not in ["Churn", "customerID"]]
cat_features

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_features, drop_first=True)

df.head()
df.shape

##################################
# MODELLING
##################################
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

# Adjusting the scale_pos_weight parameter because the dataset is imbalanced
negative_class_count = sum(y == 0)
positive_class_count = sum(y == 1)
scale_pos_weight = negative_class_count / positive_class_count
# 2.7683253076511503

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

##################################
# XGBOOST - OPTUNA
##################################

# We will optimize the F1 score using Optuna.

#def objective(trial):
#    params = {
#        'objective': 'binary:logistic',
#        'eval_metric': 'logloss',
#        'use_label_encoder': False,
#        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
#        'max_depth': trial.suggest_int('max_depth', 3, 9),
#        'learning_rate': trial.suggest_loguniform('learning_rate', 0.05, 0.3),
#        'subsample': trial.suggest_uniform('subsample', 0.2, 0.8),
#        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
#        'reg_alpha': trial.suggest_loguniform('reg_alpha', 2, 20),
#        'reg_lambda': trial.suggest_loguniform('reg_lambda', 2, 20),
#        "min_child_weight" : trial.suggest_loguniform('min_child_weight', 5, 15),
#        'scale_pos_weight': scale_pos_weight,
#        "random_state" : SEED
#    }
#    model = XGBClassifier(**params)
#    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#    f1_scores = []
#    for train_index, valid_index in skf.split(X, y):
#        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
#        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
#        model.fit(X_train, y_train)
#        y_pred = model.predict(X_valid)
#        f1_scores.append(f1_score(y_valid, y_pred))
#    return np.mean(f1_scores)

#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=75)
#best_params = study.best_params

best_params = {'n_estimators': 81,
  'max_depth': 3,
  'learning_rate': 0.12459489853975102,
  'subsample': 0.5010646040949962,
  'colsample_bytree': 0.7858790406750775,
  'reg_alpha': 11.548242910899093,
  'reg_lambda': 16.966231089295302,
  'min_child_weight': 12.960072098176274}


###################################
# XGBOOST - Stratıfıed Kfold Validation
###################################
n_splits = 5
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_model_xgb = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss',
                               objective='binary:logistic', scale_pos_weight=scale_pos_weight, random_state=SEED)
fold_precisions = []
fold_f1s = []
fold_accuracies = []
fold_roc_aucs = []
fold_recalls = []

fold_precisions_train = []
fold_f1s_train = []
fold_accuracies_train = []
fold_roc_aucs_train = []
fold_recalls_train = []

for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
    # Splitting into training and test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Training the model
    best_model_xgb.fit(X_train, y_train)

    # Making predictions
    y_pred = best_model_xgb.predict(X_test)
    y_pred_train = best_model_xgb.predict(X_train)
    y_pred_proba = best_model_xgb.predict_proba(X_test)[:, 1]
    y_pred_proba_train = best_model_xgb.predict_proba(X_train)[:, 1]

    # Calculating metrics for the test set
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)

    # Calculating metrics for the training set
    precision_train = precision_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_proba_train)
    recall_train = recall_score(y_train, y_pred_train)

    # Adding results to the lists
    fold_precisions.append(precision)
    fold_f1s.append(f1)
    fold_accuracies.append(accuracy)
    fold_roc_aucs.append(roc_auc)
    fold_recalls.append(recall)

    fold_precisions_train.append(precision_train)
    fold_f1s_train.append(f1_train)
    fold_accuracies_train.append(accuracy_train)
    fold_roc_aucs_train.append(roc_auc_train)
    fold_recalls_train.append(recall_train)

    # Printing results for each fold
    print(
        f"Fold {fold + 1} - Train: Precision={precision_train:.4f}, F1={f1_train:.4f}, Accuracy={accuracy_train:.4f}, ROC AUC={roc_auc_train:.4f}, Recall={recall_train:.4f}")
    print(
        f"          - Test: Precision={precision:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, ROC AUC={roc_auc:.4f}, Recall={recall:.4f}")

# Printing average results
print("\nMean Train: Precision={:.4f}, F1={:.4f}, Accuracy={:.4f}, ROC AUC={:.4f}, Recall={:.4f}".format(
    np.mean(fold_precisions_train), np.mean(fold_f1s_train), np.mean(fold_accuracies_train),
    np.mean(fold_roc_aucs_train), np.mean(fold_recalls_train)))

print("Mean Test: Precision={:.4f}, F1={:.4f}, Accuracy={:.4f}, ROC AUC={:.4f}, Recall={:.4f}".format(
    np.mean(fold_precisions), np.mean(fold_f1s), np.mean(fold_accuracies), np.mean(fold_roc_aucs),
    np.mean(fold_recalls)))


# Mean Train: Precision=0.5378, F1=0.6528, Accuracy=0.7656, ROC AUC=0.8646, Recall=0.8301
# Mean Test: Precision=0.5245, F1=0.6361, Accuracy=0.7546, ROC AUC=0.8461, Recall=0.8079


##################################
# CATBOOST - OPTUNA
##################################

#def objective(trial):
#    params = {
#        'iterations': trial.suggest_int('iterations', 100, 1000),
#        'depth': trial.suggest_int('depth', 4, 10),
#        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
#        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 20),
#        'border_count': trial.suggest_int('border_count', 32, 255),
#        'scale_pos_weight': scale_pos_weight,
#        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.3, 1.0),
#        'random_strength': trial.suggest_float('random_strength', 2, 15),
#        'verbose': 0
#    }
#    model = CatBoostClassifier(**params)
#    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#    f1_scores = []
#    for train_index, valid_index in skf.split(X, y):
#        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
#        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
#        model.fit(X_train, y_train)
#        y_pred = model.predict(X_valid)
#        f1_scores.append(f1_score(y_valid, y_pred))
#    return np.mean(f1_scores)

#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=25)
#best_params = study.best_params

best_params = {'iterations': 303,
 'depth': 4,
 'learning_rate': 0.06205635728927092,
 'l2_leaf_reg': 3.5872599600325383,
 'border_count': 229,
 'bagging_temperature': 0.7783264420926188,
 'random_strength': 11.670072666600827}


###################################
# CATBOOST - Stratıfıed Kfold Validation
###################################
n_splits = 5
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_model_catboost = CatBoostClassifier(**best_params, scale_pos_weight=scale_pos_weight, verbose=0,
                                         random_state=SEED)

fold_precisions = []
fold_f1s = []
fold_accuracies = []
fold_roc_aucs = []
fold_recalls = []

fold_precisions_train = []
fold_f1s_train = []
fold_accuracies_train = []
fold_roc_aucs_train = []
fold_recalls_train = []

for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
    # Splitting into training and test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Training the model
    best_model_catboost.fit(X_train, y_train)

    # Making predictions
    y_pred = best_model_catboost.predict(X_test)
    y_pred_train = best_model_catboost.predict(X_train)
    y_pred_proba = best_model_catboost.predict_proba(X_test)[:, 1]
    y_pred_proba_train = best_model_catboost.predict_proba(X_train)[:, 1]

    # Calculating metrics for the test set
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)

    # Calculating metrics for the training set
    precision_train = precision_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_proba_train)
    recall_train = recall_score(y_train, y_pred_train)

    # Adding results to the lists
    fold_precisions.append(precision)
    fold_f1s.append(f1)
    fold_accuracies.append(accuracy)
    fold_roc_aucs.append(roc_auc)
    fold_recalls.append(recall)

    fold_precisions_train.append(precision_train)
    fold_f1s_train.append(f1_train)
    fold_accuracies_train.append(accuracy_train)
    fold_roc_aucs_train.append(roc_auc_train)
    fold_recalls_train.append(recall_train)

    # Printing results for each fold
    print(
        f"Fold {fold + 1} - Train: Precision={precision_train:.4f}, F1={f1_train:.4f}, Accuracy={accuracy_train:.4f}, ROC AUC={roc_auc_train:.4f}, Recall={recall_train:.4f}")
    print(
        f"          - Test: Precision={precision:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, ROC AUC={roc_auc:.4f}, Recall={recall:.4f}")

# Printing average results
print("\nMean Train: Precision={:.4f}, F1={:.4f}, Accuracy={:.4f}, ROC AUC={:.4f}, Recall={:.4f}".format(
    np.mean(fold_precisions_train), np.mean(fold_f1s_train), np.mean(fold_accuracies_train),
    np.mean(fold_roc_aucs_train), np.mean(fold_recalls_train)))

print("Mean Test: Precision={:.4f}, F1={:.4f}, Accuracy={:.4f}, ROC AUC={:.4f}, Recall={:.4f}".format(
    np.mean(fold_precisions), np.mean(fold_f1s), np.mean(fold_accuracies), np.mean(fold_roc_aucs),
    np.mean(fold_recalls)))

# Mean Train: Precision=0.5634, F1=0.6789, Accuracy=0.7856, ROC AUC=0.8856, Recall=0.8539
# Mean Test: Precision=0.5280, F1=0.6348, Accuracy=0.7569, ROC AUC=0.8462, Recall=0.7961

################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(best_model_xgb, X)
plot_importance(best_model_catboost, X)

##################################
# LOGISTIC REGRESSION
##################################

# Set to k = 5 using Stratified KFold
n_splits = 5
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# call the SMOTE function for oversampling
oversample = SMOTE(random_state=77)

# logistic regression model
model_logistic = LogisticRegression()

# Lists to store the results
fold_precisions = []
fold_f1s = []
fold_accuracies = []
fold_roc_aucs = []
fold_recalls = []

# Lists to store the results of the training data
fold_precisions_train = []
fold_f1s_train = []
fold_accuracies_train = []
fold_roc_aucs_train = []
fold_recalls_train = []

# Validation for each fold
for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
    # Splitting into training and test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_smote, y_smote = oversample.fit_resample(X_train, y_train)

    # Training the model
    model_logistic.fit(X_smote, y_smote)

    # Make predictions
    y_pred = model_logistic.predict(X_test)
    y_pred_train = model_logistic.predict(X_train)
    y_pred_proba = model_logistic.predict_proba(X_test)[:, 1]
    y_pred_proba_train = model_logistic.predict_proba(X_train)[:, 1]

    # Calculating metrics for the test set
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)

    # Calculating metrics for the training set
    precision_train = precision_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_proba_train)
    recall_train = recall_score(y_train, y_pred_train)

    # Adding results to the lists
    fold_precisions.append(precision)
    fold_f1s.append(f1)
    fold_accuracies.append(accuracy)
    fold_roc_aucs.append(roc_auc)
    fold_recalls.append(recall)

    fold_precisions_train.append(precision_train)
    fold_f1s_train.append(f1_train)
    fold_accuracies_train.append(accuracy_train)
    fold_roc_aucs_train.append(roc_auc_train)
    fold_recalls_train.append(recall_train)

    # Printing results for each fold
    print(
        f"Fold {fold + 1} - Train: Precision={precision_train:.4f}, F1={f1_train:.4f}, Accuracy={accuracy_train:.4f}, ROC AUC={roc_auc_train:.4f}, Recall={recall_train:.4f}")
    print(
        f"          - Test: Precision={precision:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}, ROC AUC={roc_auc:.4f}, Recall={recall:.4f}")

# Printing average results
print("\nMean Train: Precision={:.4f}, F1={:.4f}, Accuracy={:.4f}, ROC AUC={:.4f}, Recall={:.4f}".format(
    np.mean(fold_precisions_train), np.mean(fold_f1s_train), np.mean(fold_accuracies_train),
    np.mean(fold_roc_aucs_train), np.mean(fold_recalls_train)))

print("Mean Test: Precision={:.4f}, F1={:.4f}, Accuracy={:.4f}, ROC AUC={:.4f}, Recall={:.4f}".format(
    np.mean(fold_precisions), np.mean(fold_f1s), np.mean(fold_accuracies), np.mean(fold_roc_aucs),
    np.mean(fold_recalls)))


# Mean Train: Precision=0.5560, F1=0.6192, Accuracy=0.7720, ROC AUC=0.8298, Recall=0.6989
# Mean Test: Precision=0.5473, F1=0.6111, Accuracy=0.7664, ROC AUC=0.8242, Recall=0.6923





