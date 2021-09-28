import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 

training_data = pd.read_csv("titanic_train.csv")
print(training_data.info())
print("")
print(training_data.describe())
print("")
# Checking to see if there is any missing data
missing_data = training_data.isnull()

# sns.heatmap(missing_data, yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')

# Creating a pair plot 
# sns.pairplot(training_data)

# Creating count plot to see how many people survived
# sns.countplot(x="Survived", hue="Pclass", data = training_data)

# Plotting the age of the passengers 
# sns.distplot(training_data["Age"].dropna(), kde = False, bins=30)

# Plotting the number of siblings 
# sns.countplot(x="SibSp", data=training_data)

# Plotting a histogram for fare 
# training_data["Fare"].hist(bins=40)

# sns.boxplot(x="Pclass", y="Age", data=training_data)

def impute_age(cols): 
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1: 
            return 37
        elif Pclass == 2: 
            return 29 
        else: 
            return 24 
    else: 
        return Age
    
training_data["Age"] = training_data[["Age", "Pclass"]].apply(impute_age, axis = 1)

# sns.heatmap(training_data.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')

# Dropping cabin from training data
training_data.drop("Cabin", axis = 1, inplace = True)

training_data.dropna(inplace = True)

# No more missing values
# sns.heatmap(training_data.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')

# Converting categorical features to a dummy variable using pandas
sex = pd.get_dummies(training_data["Sex"], drop_first = True)
embark = pd.get_dummies(training_data["Embarked"], drop_first = True)

# Adding the dummy variables in the dataframe
training_data = pd.concat([training_data, sex, embark], axis = 1)

training_data.drop(["Sex", "Embarked", "Name", "Ticket", "PassengerId"], axis = 1, inplace = True)

X = training_data.drop("Survived", axis = 1)
y = training_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))