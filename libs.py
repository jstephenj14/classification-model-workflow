# libraries and preset options
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix as CM 
from sklearn import model_selection 
from sklearn.model_selection import learning_curve 
from sklearn.metrics import precision_recall_curve 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

%matplotlib inline
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 
sns.pairplot(data)
data.hist(figsize=(20,10))
data.groupby("dept")["salary"].hist(alpha=0.9, legend=True)
sns.pairplot(conversion_data[["age","total_pages_visited","converted"]], hue="converted",height = 10)

# 
creditcard.hist(figsize=(15,15))

limit_dict = {}
for col in creditcard.columns:
    ul = input(f"enter UPPER LIMIT for {col}:")
    if ul == "skip":
        continue
    ll =input(f"enter LOWER LIMIT for {col}:")
    limit_dict[col] = [float(ll), float(ul)]
    
df = creditcard.copy()
for col in df.columns:
    if col in limit_dict.keys():
        df = df[(df[col] > limit_dict[col][0]) & (df[col] < limit_dict[col][1])]
        print(col)
        print(df.shape)
        
#
target = "class"
predictors = [c for c in creditcard.columns if c != "class"]

# 
X = df[predictors].values
y = df[target].values

model = XGBClassifier(learning_rate = 0.01, n_estimators=2000, 
                        max_depth=20, objective = "binary:logistic")
over = SMOTE(sampling_strategy=0.1, k_neighbors=5)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]

pipeline = Pipeline(steps=steps)
cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)
scores_over = cross_val_score(pipeline, X, y, scoring='recall', cv=cv, n_jobs=-1)
print(f"k={k}\n")
print(f"mean recall: {np.mean(scores_over)}\n")
print(scores_over)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
pipeline.fit(X_train,y_train)

yhat_test = pipeline.predict(X_test)
yhat_test_proba = pipeline.predict_proba(X_test)[:,1]

confusion_matrix = CM(y_test,yhat_test,np.unique(y_train))

precision_ls, recall_ls, threshold_ls =  precision_recall_curve(y_test,yhat_test_proba)

plt.figure(figsize=(10,10))
threshold_ls = np.append(threshold_ls,1)
plt.plot(threshold_ls, precision_ls)
plt.plot(threshold_ls, recall_ls)
plt.legend(["precision","recall"])

tree1 = DecisionTreeClassifier( max_depth=3, min_samples_leaf = 30, class_weight="balanced")
tree1.fit(X_train, y_train)

fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(tree1, 
                   feature_names=predictors,  
                   filled=True)

feature_imp_df = pd.DataFrame({"predictor":predictors,"feature_imp":pipeline["model"].feature_importances_})
feature_imp_df.sort_values("feature_imp",ascending=False)

