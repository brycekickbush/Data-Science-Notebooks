# project: p7
# submitter: kickbush
# partner: ------
# hours: 5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

class UserPredictor:
    def __init__(self):
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('std', StandardScaler()),
            ('lr', LogisticRegression())
       
        ])
        self.xcols= ['user_id', 'past_purchase_amt', 'age', 'total_seconds']

        self.userTest = None
    def fit(self, train_users, train_logs, train_y):
        #Create a dictionary with user_id starting from 0-last id, if they have seconds add them, if not set them at 0 (never visited site)
        secondDict = train_logs.groupby('user_id')['seconds'].apply(list).to_dict()
        for i in range(300000):
            if i in secondDict:
                secondDict[i] = sum(secondDict[i])
            else:
                secondDict[i] = 0
        self.secondDict = secondDict
        train_users['total_seconds'] = train_users['user_id'].map(secondDict)
        train_users['y'] = train_y['y']
        train, test = train_test_split(train_users, random_state = 320)
        self.model.fit(train[self.xcols], train['y'])
        print(self.model.score(test[self.xcols], test['y']))
            
    def predict(self, train_users, train_logs):
        newDict = train_logs.groupby('user_id')['seconds'].apply(list).to_dict()
        for i in range(300000):
            if i in newDict:
                newDict[i] = sum(newDict[i])
            else:
                newDict[i] = 0
        train_users['total_seconds'] = train_users['user_id'].map(newDict)
        train_users['predicted'] = self.model.predict(train_users[self.xcols])
        listPredict = train_users["predicted"].tolist()
        return np.array(listPredict)