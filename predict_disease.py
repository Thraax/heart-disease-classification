# Import the tools


# EDA tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Machine learning models

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Model evaluations

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import r2_score

# ____ Explore our Data ______

path = 'Data path'
heart_disease = pd.read_csv(path)
heart_disease.shape  # (rows, columns)

# Take a look on the data
heart_disease.head()

# Explore the target value
heart_disease['target'].value_counts()

# Visualize the ratio of the (1,0) in the target column
target_vis = heart_disease['target'].value_counts()
target_vis.plot(kind='bar', xlabel='Target label', ylabel='Numbers of target', title='Positive VS Negative labels')
plt.show()

"""
Personally i prefer using pie chart when we have small number of labels 
We will try this because we talk about ratio, wanna see if there one label dominate on the others.
"""

# Try visual the target with pie chart
target_vis = heart_disease['target'].value_counts()
target_vis.plot(kind='pie', title='Positive VS Negative labels')
plt.show()

# Looking for missing values

"""
I am usually doing that by looking at the ratio of missing from column not the sum of the nulls
that's because the ratio give you good picture of how many of the data lost from certain column
The sum is just a solid number need to abstract the total from it to get better picture
"""

missing_ratio = pd.concat([heart_disease.isnull().sum(),
                           heart_disease.isnull().sum() / heart_disease.isnull().count()],
                          axis=1, keys=['Sum', 'Percentage']).sort_values(ascending=False, by=['Sum'])

"""
Seems we are lucky, after printing the missing ratio there is no nulls in our dataset
"""

# Heart disease frequency according to sex

pd.crosstab(heart_disease.sex, heart_disease.target)

# Visualize the frequency

disease_sex_freq = pd.crosstab(heart_disease.target, heart_disease.sex)

# Rename the columns and the indices to be more readable
disease_sex_freq = disease_sex_freq.rename(columns={0: 'Female', 1: 'Male'},
                                           index={0: 'Negative case', 1: 'Positive case'})

disease_sex_freq.plot(kind='bar', xlabel='Sex', ylabel='Sex frequency',
                      title='Heart disease frequency according to sex')
plt.xticks(rotation=0)
plt.show()

# Take a look at the age
sns.boxplot(heart_disease.age)  # the median value of ages around 50 - 60
sns.distplot(heart_disease.age)  # look like semi-normal shape

# Is there a relation between age and cholesterol?
sns.scatterplot(x=heart_disease.age, y=heart_disease.chol, data=heart_disease)  # Seems there relation but very weak
# measure the correlation numerically
heart_disease['age'].corr(heart_disease['chol'])  # 0.213, No power correlation between them

# Let's take a close look on age and cholesterol with positive cases and negative cases separately
sns.relplot(x=heart_disease.age, y=heart_disease.chol, data=heart_disease, kind='scatter',
            col=heart_disease.target, hue=heart_disease.sex)

# measure the correlation between age and cholesterol with positive cases numerically
pos_cases = heart_disease[heart_disease.target == 1]
pos_cases['age'].corr(pos_cases['chol'])  # pretty high, but not that much

# Measure the correlation between age and max heart rate 'thalach'

# Start by measuring the correlation in all cases (positive/Negative), then separate them
sns.scatterplot(heart_disease.age, heart_disease.thalach, color='red')  # In general it's look negative correlation

# Separate the scatter by cases labels
sns.relplot(x=heart_disease.age, y=heart_disease.thalach, data=heart_disease, kind='scatter',
            col=heart_disease.target, hue=heart_disease.sex)

"""
From the scatter plot we can see that:
 * Both labels have negative correlation with (age, max heart rate) 
 * When the case has negative label the correlation became more weakly than the positive one
"""

# Measure the correlations numerically

# Correlation in general:
heart_disease['age'].corr(heart_disease.thalach)  # -0.39

# Correlation with positive cases:
pos_cases['age'].corr(heart_disease.thalach)  # -0.52

# Build the correlation matrix to take the big picture
corr_matrix = heart_disease.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f")

# Build the model

X = heart_disease.drop('target', axis=1)
y = heart_disease.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Put the models into dictionary
models = {"Logistic regression": LogisticRegression(max_iter=2000), "KNN": KNeighborsClassifier(n_neighbors=3),
          "SVM": SVC(),
          'Random Forest': RandomForestClassifier(n_estimators=1000),
          "Decision Tree": DecisionTreeClassifier(criterion="entropy")}


# Make function that test and give the score for us

def test_and_score(Models, Xtrain, ytrain, Xtest, ytest):
    """
    :param Models:  This is a dictionary containing the models that will be fit the data
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test:  Test data
    :param y_test:  Test labels
    :return:        return dictionary of the models and their scores
    """

    model_scores = {}

    # Loop on the models
    for name, model in Models.items():
        # Fit the model to the data
        model.fit(Xtrain, ytrain)
        # Evaluate the model
        model_scores[name] = model.score(Xtest, ytest)

    return model_scores


models_scores = test_and_score(models, X_train, y_train, X_test, y_test)

models_compare = pd.DataFrame(models_scores, index=['Accuracy'])
models_compare.T.plot.bar()
plt.xticks(rotation=-10)

"""
'Logistic regression': 0.9016393442622951,
 'KNN': 0.7213114754098361,
 'SVM': 0.6721311475409836,
 'Random Forest': 0.9016393442622951,
 'Decision Tree': 0.7868852459016393
"""

# Let's give the KNN and SVM other chance and tune their hyper_parameters


# Loop on KNN n_neighbors parameter

KNN_train_scores = []
KNN_test_scores = []

for i in range(1, 21):
    KNN = KNeighborsClassifier(n_neighbors=i)
    # Fit the model
    KNN.fit(X_train, y_train)
    # Append the score
    KNN_train_scores.append(KNN.score(X_test, y_test))
    KNN_test_scores.append(KNN.score(X_train, y_train))

# Plot the results
sns.lineplot(range(1, 21), KNN_train_scores, label='Train score', color='red')
sns.lineplot(range(1, 21), KNN_test_scores, label='Test score', color='blue')
plt.xlabel('NO. Neighbours')
plt.ylabel('Score')
plt.title("Number of neighbours VS Score")
plt.legend()
plt.show()

KNN = KNeighborsClassifier(n_neighbors=15)
# Fit the model
KNN.fit(X_train, y_train)
# Get the score
KNN.score(X_test, y_test)  # 0.68, Still awful

# Let's try tune the SVC
sp_vec = SVC(kernel='linear', degree=6, C=4.9995)
sp_vec.fit(X_train, y_train)
sp_vec.score(X_test, y_test)  # 0.86, now he is competitive one!

# Tuning the hyper parameters with RandomizedSearchCV (Logistic regression - RandomForest - SVC)

logistic_CV = {"C": np.logspace(-4, 4, 20), "solver": ['liblinear', 'lbfgs'], "max_iter": np.arange(100, 2000, 100)}

randomfor_CV = {"n_estimators": np.arange(100, 1000, 50), "max_depth": [None, 3, 5, 10],
                "min_samples_split": np.arange(2, 20, 2), "min_samples_leaf": np.arange(1, 20, 2)}

SVC_CV = {"C": np.arange(0.5, 20, 1), 'kernel': ['rbf', 'linear'], 'degree': np.arange(1, 16, 1)}

# Setup the RandomizedSearchCV
RSCV_logistic = RandomizedSearchCV(LogisticRegression(), param_distributions=logistic_CV,
                                   cv=5, n_iter=20, verbose=True)

RSCV_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=randomfor_CV,
                             cv=5, n_iter=20, verbose=True)

RSCV_SVC = RandomizedSearchCV(SVC(), param_distributions=SVC_CV, cv=5, n_iter=20, verbose=True)

# Fit the models with the new parameters
RSCV_logistic.fit(X_train, y_train)
RSCV_rf.fit(X_train, y_train)
RSCV_SVC.fit(X_train, y_train)

# Test the scores
RSCV_rf.score(X_test, y_test)
RSCV_logistic.score(X_test, y_test)
RSCV_SVC.score(X_test, y_test)

# Setup the GridSearchCV and see if there is any difference (SVC - Logistic regression)

GS_SVC = GridSearchCV(SVC(), param_grid=SVC_CV, cv=5, verbose=True)
GS_logistic = GridSearchCV(LogisticRegression(), param_grid=logistic_CV, cv=5, verbose=True)

# Fit the models
GS_SVC.fit(X_train, y_train)
GS_logistic.fit(X_train, y_train)

# Get the models scores
GS_SVC.score(X_test, y_test)
GS_logistic.score(X_test, y_test)

# Evaluate the model

y_pred = GS_logistic.predict(X_test)

# Plotting the ROC curve
plot_roc_curve(GS_logistic, X_test, y_test)

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(5, 5))
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.title("Predicted labels VS True labels")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cbar=True)
plt.show()

# Evaluate the model with the Cross-Validation

# Create classifier with best Hyper-parameters
GS_logistic.best_params_
clf = LogisticRegression(C=0.08858667904100823, max_iter=400, solver='lbfgs')
clf.fit(X_train, y_train)

# Cross validation accuracy
cv_accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
np.mean(cv_accuracy)

# Cross validation F1 score
cv_F1 = cross_val_score(clf, X, y, cv=5, scoring='f1')
np.mean(cv_F1)


