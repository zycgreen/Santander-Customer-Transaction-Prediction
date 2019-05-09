#!/usr/bin/env python
# coding: utf-8

# In[6]:


# supervised imbalanced classification case
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from termcolor import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import imblearn
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost
import warnings
warnings.filterwarnings('ignore')


# In[7]:


os.getcwd()


# In[8]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[9]:


train_df = train.copy()
test_df = test.copy()
train_df.head()


# In[10]:


test_df.head()


# In[11]:


train_df.shape


# In[12]:


test_df.shape


# In[13]:


train_df.columns


# In[14]:


# may need feature scaling
train_df.describe()


# In[15]:


test_df.describe()


# In[16]:


def check_missing(data):
    # Total missing values
    mis = data.isnull().sum()
    
    # percentage of missing values
    mis_p = 100 * mis / len(data)
    
    # make a table with the results
    mis_t = pd.concat([mis,mis_p],axis = 1)
    
    # Renamw the columns
    mis_tt = mis_t.rename(columns = {0:'Missing Values',
                                    1: '% of Total Values'})
    # sort
    mis_tt = mis_tt[mis_tt.iloc[:,1] != 0].sort_values('% of Total Values', ascending = False).round(1)
    
    return mis_tt


# In[17]:


check_missing(train_df) # no missing value


# In[18]:


# All attributes are numerical
train_df.info()


# In[19]:


test_df.info()


# In[20]:


train_df.iloc[:,2:].hist(bins=50, figsize = (20,15))
# nearly normal distributed


# In[21]:


sns.countplot(train_df['target'], palette = "Set2")
# there is an imbalanced class problem


# In[22]:


train_df['target'].value_counts()


# In[23]:


features = train_df.columns.values[2:203]
correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.tail(10) # correlation between each variable is not significance


# In[26]:


# try logistic regression -- why imbalanced will be a bring problem. 

# Split 70% train and 30% test set
X, y = train_df.iloc[:,2:].values, train_df.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                stratify = y,
                random_state = 0)

# Standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

lr_reg = LogisticRegression()
clf = lr_reg.fit(X_train_std, y_train)
pred_lr = clf.predict(X_test_std)


# In[27]:


# Accuracy = TP+TN/Total
# Precison = TP/(TP+FP)
# Recall = TP/(TP+FN)
# TP = True possitive means no. of possitve cases which are predicted possitive
# TN = True negative means no. of negative cases which are predicted negative
# FP = False possitve means no. of negative cases which are predicted possitive
# FN= False Negative means no. of possitive cases which are predicted negative
# False Negative Rate: the closer to 0, the better. FN/(FN + TP)
# True Positve Rate: the closer to 1, the better. TP/(FP + TP)
# True Negative Rate: the closer to 1, the better. TN/(TN + FN)
# False Positve Rate: the closer to 0, the better. FP/(FP + TN)

lr_matrix = confusion_matrix(y_test, pred_lr)
print("Recall :",lr_matrix[1,1]/(lr_matrix[1,1]+lr_matrix[1,0]))
print("Accuracy :",(lr_matrix[1,1]+lr_matrix[0,0])/(lr_matrix[1,1]+lr_matrix[0,1]+lr_matrix[0,0]+lr_matrix[1,0]))
fig = plt.figure(figsize=(9,6))
print("TP",lr_matrix[1,1],":","no. of 1 which are predicted 1") 
print("TN",lr_matrix[0,0],":","no. of 0 which are predicted 0") 
print("FP",lr_matrix[0,1],":","no. of 0 which are predicted 1") 
print("FN",lr_matrix[1,0],":","no. of 1 which are predicted 0")
print("FN rate :",lr_matrix[1,0]/(lr_matrix[1,0]+lr_matrix[1,1]))
sns.heatmap(lr_matrix,cmap="BuPu",annot=True,linewidths=0.5)
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()
print("\n--------------------------------Classification Report------------------------------------")

# If the classifier’s performance is determined by the number of mistakes, then clearly lr model is good.

# However, if our goal is to predict class 1 more precisely, then maybe lr model is not very good, its FNR is 0.72.
# High accuracy rate was just an illusion.


# In[28]:


# try PCA

# Covar matrix / eigenvalues / eigenvectors
cov_mat = np.cov(X_train_std.T)
eigen_va, eigen_vc = np.linalg.eig(cov_mat)
tot = sum(eigen_va)
var_exp = [(i / tot) for i in
          sorted(eigen_va, reverse = True)]
cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize = (8,5))
plt.bar(range(0,200),var_exp, alpha = 0.5, align = 'center',
       label = 'individual explained variance')
plt.step(range(0,200),cum_var_exp, where = 'mid',
        label = 'cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc = 'best')
plt.show()

# From the plot, we can see PCA is not very useful, since the correlation between features is not significance.


# In[29]:


# Exploratory Data Analysis
def plot_density(df_1,df_2,features):
    i = 0
    fig , ax = plt.subplots(6,6,figsize=(12,10))
    for feature in features:
        i += 1
        plt.subplot(6,6,i)
        sns.kdeplot(df_1[feature],shade = True, color = "r",label = str(0) + " " + features[i-1])
        sns.kdeplot(df_2[feature],shade = True, color = "c",label = str(1) + " " + features[i-1])
    plt.show()


# In[30]:


sns.set_style("darkgrid")
t0 = train_df.loc[train_df['target']==0]
t1 = train_df.loc[train_df['target']==1]
features_1 = train_df.columns.values[2:38]
plot_density(t0,t1,features_1)


# In[31]:


features_2 = train_df.columns.values[38:74]
plot_density(t0,t1,features_2)


# In[32]:


features_3 = train_df.columns.values[74:110]
plot_density(t0,t1,features_3)


# In[33]:


features_4 = train_df.columns.values[110:146]
plot_density(t0,t1,features_4)


# In[34]:


features_5 = train_df.columns.values[146:182]
plot_density(t0,t1,features_5)


# In[35]:


features_6 = train_df.columns.values[182:202]
plot_density(t0,t1,features_6)

# some variables distributed differently in each category. like var_0, var_1, var_2
# some seem distributed likely. like var_4, var_10, var_14


# In[36]:


correlation = train_df.corr()['target'].abs().sort_values()
correlation.tail(10)


# In[37]:


# mean
train_n = train_df.iloc[:,2:203]
t0_n = t0.iloc[:,2:203]
t1_n = t1.iloc[:,2:203]
train_nn = train_n.mean(axis = 1) # per row
train_cn = train_n.mean(axis = 0) # per column
t0_nn = t0_n.mean(axis = 1)
t1_nn = t1_n.mean(axis = 1)
t0_cn = t0_n.mean(axis = 0)
t1_cn = t1_n.mean(axis = 0)


# In[38]:


sns.distplot(train_nn,bins = 100,kde = True, color = 'darkorange', label = 'train')
plt.title('per row')
plt.legend()


# In[39]:


sns.distplot(train_cn,bins = 100,kde = True, color = 'darkorange', label = 'train')
plt.title('per column')
plt.legend()


# In[40]:


sns.distplot(t0_nn,bins = 100,kde = True, color = 'darkblue', label = 'train target = 0')
sns.distplot(t1_nn,bins = 100,kde = True, color = 'darkorange', label = 'train target = 1')
plt.title("row  train target = 0 & train target = 1")
plt.legend()


# In[41]:


sns.distplot(t0_cn,bins = 100,kde = True, color = 'darkblue', label = 'train target = 0')
sns.distplot(t1_cn,bins = 100,kde = True, color = 'darkorange', label = 'train target = 1')
plt.title("column  train target = 0 & train target = 1")
plt.legend()


# In[42]:


# std
train_nn_s = train_n.std(axis = 1) # per row
train_cn_s = train_n.std(axis = 0) # per column
t0_nn_s = t0_n.std(axis = 1)
t1_nn_s = t1_n.std(axis = 1)
t0_cn_s = t0_n.std(axis = 0)
t1_cn_s = t1_n.std(axis = 0)
sns.set_style("darkgrid")


# In[43]:


sns.distplot(train_nn_s,bins = 100,kde = True, color = 'darkorange', label = 'train')
plt.title('std per row')
plt.legend()


# In[44]:


sns.distplot(train_cn_s,bins = 100,kde = True, color = 'darkorange', label = 'train')
plt.title('std per column')
plt.legend()


# In[45]:


sns.distplot(t0_nn_s,bins = 100,kde = True, color = 'darkblue', label = 'train target = 0')
sns.distplot(t1_nn_s,bins = 100,kde = True, color = 'darkorange', label = 'train target = 1')
plt.title("std row  train target = 0 & train target = 1")
plt.legend()


# In[46]:


sns.distplot(t0_cn_s,bins = 100,kde = True, color = 'darkblue', label = 'train target = 0')
sns.distplot(t1_cn_s,bins = 100,kde = True, color = 'darkorange', label = 'train target = 1')
plt.title("std column  train target = 0 & train target = 1")
plt.legend()

# fram above plots, we can see we need to select important features or constuct new features.


# In[47]:


# features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
# target = train_df['target']


# In[49]:


# LightGBM -- no Standardization (tree based model)
params = {
    'objective': 'binary', 
    'boosting_type': 'gbdt',
    'is_unbalance': 'true',
    'metric':'auc',
    'learning_rate': 0.01,
    'bagging_freq': 3,
    'bagging_fraction': 0.5,
    'boost_from_average':'false',
    'feature_fraction': 0.05,
    'max_depth': -1,  # no limit
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 5,
    'tree_learner': 'serial',
    'verbosity': 1
}

target = 'target'
predictors = train_df.columns.values.tolist()[2:]

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2019)
pred = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()
fold_importance_df = pd.DataFrame()

for fold_,(tr_id, val_id) in enumerate(skf.split(train_df, train_df.target.values)):
    print("\nfold {}".format(fold_))
    dtrain = lgb.Dataset(train_df.iloc[tr_id][predictors].values,
                        label = train_df.iloc[tr_id][target].values,
                        feature_name = predictors,
                        free_raw_data = False)
    dvalid = lgb.Dataset(train_df.iloc[val_id][predictors].values,
                        label = train_df.iloc[val_id][target].values,
                        feature_name = predictors,
                        free_raw_data = False)
    nround = 10000
    clf = lgb.train(params,
                    dtrain,
                    nround,
                    valid_sets = dvalid,
                    valid_names = 'valid',
                    verbose_eval = 250,
                    early_stopping_rounds = 100)
   
    pred[val_id] = clf.predict(train_df.iloc[val_id][predictors].values, 
                               num_iteration = clf.best_iteration)  ## [val_id] - index
    
    fold_importance_df["predictors"] = predictors
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df,fold_importance_df],axis = 0)
    
    predictions += clf.predict(test_df[predictors], num_iteration = clf.best_iteration) / 10
    
    print("\n\nCV AUC: {:<0.5f}".format(metrics.roc_auc_score(train_df.target.values, pred)))


# In[50]:


lgbm_matrix = confusion_matrix(train_df.target.values, pred.round())
print("Recall :",lgbm_matrix[1,1]/(lgbm_matrix[1,1]+lgbm_matrix[1,0]))
print("Accuracy :",(lgbm_matrix[1,1]+lgbm_matrix[0,0])/(lgbm_matrix[1,1]+lgbm_matrix[0,1]+lgbm_matrix[0,0]+lgbm_matrix[1,0]))
fig = plt.figure(figsize=(9,6))
print("TP",lgbm_matrix[1,1],":","no. of 1 which are predicted 1") 
print("TN",lgbm_matrix[0,0],":","no. of 0 which are predicted 0") 
print("FP",lgbm_matrix[0,1],":","no. of 0 which are predicted 1") 
print("FN",lgbm_matrix[1,0],":","no. of 1 which are predicted 0")
print("FN rate :",lgbm_matrix[1,0]/(lgbm_matrix[1,0]+lgbm_matrix[1,1]))
sns.heatmap(lgbm_matrix,cmap="BuPu",annot=True,linewidths=0.5)
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()
print("\n--------------------------------Classification Report------------------------------------")


# In[51]:


feature_importance_col = (feature_importance_df[["predictors","importance"]]
                          .groupby("predictors")
                          .mean()
                          .sort_values(by = "importance", ascending = False).index)
best_features = feature_importance_df.loc[feature_importance_df.predictors.isin(feature_importance_col)]

plt.figure(figsize = (14,26))
sns.barplot(x = "importance", y = "predictors", 
            data = best_features.sort_values(by = "importance", ascending = False))
plt.title('LightBGM Features (averaged over folds)')
plt.tight_layout()


# In[52]:


# Use Lightgbm as a feature selection tool
best_f = best_features.groupby('predictors')['importance'].mean().sort_values(ascending = False)

j = 0
for i in best_f.index:
    if best_f[i] == 0:
        j += 1
        print(str(j) + ": " + i + " " + str(0))       


# In[53]:


best_ff = best_f/best_f.sum()  # sum to 1
best_ff[:170].sum() 


# In[127]:


sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)


# In[47]:


# f_index = best_ff[:120].index
# final_data = pd.DataFrame(train_df,columns = f_index) # select certain columns


# In[48]:


# X_train_std_n = pd.DataFrame(X_train_std)
# X_train_std_n.columns = predictors
# X_train_std_new = pd.DataFrame(X_train_std_n,columns = f_index) # select certain columns

# X_test_std_n = pd.DataFrame(X_test_std)
# X_test_std_n.columns = predictors
# X_test_std_new = pd.DataFrame(X_test_std_n,columns = f_index) # select certain columns


# In[49]:


print("--------------------------mitigate Class Imbalance Problem------------------------------\n")
print(colored("Cost Function Based Approaches: ","blue"))
print("one false negative is worse than one false positive, \nwe will count that one false negative as, e.g., 100 false negatives instead.")

print(colored("\nSampling Based Approaches: ","blue"))
print("1.Oversampling, by adding more of the minority class so it has more effect \non the machine learning algorithm")
print(colored("weakness: ","red"),"overfitting")
print(colored("meathods: ","green"),"RandomOverSampler","SMOTE")

print("\n2.Undersampling, by removing some of the majority class so it has less \neffect on the machine learning algorithm")
print(colored("weakness: ", "red"),"may lose useful imformation")
print(colored("meathods: ","green"),"RandomUnderSampler","TomekLinks","ClusterCentroids")

print("\n3.Hybrid, a mix of oversampling and undersampling")
print(colored("weakness: ", "red"),"trade-off")
print(colored("meathods: ","green"),"SMOTETomek")


# In[68]:


# SMOTE (Synthetic Minority Oversaample Technique)
# smote = SMOTE(ratio = 'minority',random_state = 2019)
# X_sm, y_sm = smote.fit_sample(X,y) -- overfitting

