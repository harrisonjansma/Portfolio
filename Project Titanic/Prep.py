import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
#Remove irrelevent features, fill NaN, reset index, set dtypes
titanic = pd.read_csv('titanic_train.csv')

#Set index
titanic = titanic.set_index('PassengerId')

#Split name into Surname and Title for family tracking
titanic['Name_Mod'] = titanic.Name.apply(lambda x: x.split())
titanic['Surname'] = titanic.Name_Mod.apply(lambda x: x[0][:-1])
titanic['Title'] = titanic.Name_Mod.apply(lambda x: x[1])
titanic = titanic.drop(['Ticket', 'Name', 'Name_Mod', 'Cabin'], axis = 1)

#dropping the Names
titanic = titanic.iloc[:,:-2]


# drop the NaN Values. (Could find a way to impute, but that is an exercise for another day)
titanic  = titanic.dropna()
titanic = pd.get_dummies(titanic)




titanic_scaled = titanic.copy()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
q = sc.fit_transform(titanic_scaled.loc[:,['Age', 'Fare']])
titanic_scaled['Age']= q[:, 0]
titanic_scaled['Fare'] = q[:,1]



from scipy import stats
titanic = titanic[(np.abs(stats.zscore(titanic[['Age','Fare']])) < 4).all(axis=1)]


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    counter = 0
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(titanic.Survived.value_counts()[counter]),
                ha='center', va='bottom')
        counter+=1
        