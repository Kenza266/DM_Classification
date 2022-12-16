import pickle
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

cat = ['OverTime',
'MaritalStatus',
'JobRole',
'Gender',
'EducationField',
'Department',
'BusinessTravel',
'Attrition']

models = {}
st.set_page_config(layout="wide")

class Node:
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    @staticmethod
    def _entropy(s):
        counts = np.bincount(np.array(s, dtype=np.int64))
        percentages = counts / len(s)

        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy
    
    def _information_gain(self, parent, left_child, right_child):
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))
    
    def _best_split(self, X, y):
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            for threshold in np.unique(X_curr):
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                if len(df_left) > 0 and len(df_right) > 0:
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]

                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        } 
                        best_info_gain = gain
        return best_split
    
    def _build(self, X, y, depth=0):
        n_rows, n_cols = X.shape
        
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            best = self._best_split(X, y)
            if best['gain'] > 0:
                left = self._build(
                    X=best['df_left'][:, :-1], 
                    y=best['df_left'][:, -1], 
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1], 
                    y=best['df_right'][:, -1], 
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'], 
                    threshold=best['threshold'], 
                    data_left=left, 
                    data_right=right, 
                    gain=best['gain']
                )
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
    
    def fit(self, X, y):
        self.root = self._build(X, y)
        
    def _predict(self, x, tree):
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)
        
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.data_right)
        
    def predict(self, X):
        return [self._predict(x, self.root) for x in X]

class RandomForest:
    def __init__(self, num_trees=5, min_samples_split=2, max_depth=5):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_trees = []
        
    @staticmethod
    def _sample(X, y):
        n_rows, n_cols = X.shape
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[samples], y[samples]
        
    def fit(self, X, y):
        if len(self.decision_trees) > 0:
            self.decision_trees = []
            
        num_built = 0
        while num_built < self.num_trees:
            try:
                clf = DecisionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth
                )
                _X, _y = self._sample(X, y)
                clf.fit(_X, _y)
                self.decision_trees.append(clf)
                num_built += 1
            except Exception as e:
                continue
    
    def predict(self, X):
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        
        y = np.swapaxes(a=y, axis1=0, axis2=1)
        
        predictions = []
        for preds in y:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        return predictions

data = pd.read_csv('datset.csv')
data_og = data.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Attrition'], axis=1)
attributes = [i for i in list(data.columns) if i not in ['Attrition']]
instance_txt = [49,'No','Travel_Frequently',279,'Research & Development',8,1,'Life Sciences',1,2,3,'Male',61,2,2,'Research Scientist',2,'Married',5130,24907,1,'Y','No',23,4,4,80,1,10,3,3,10,7,1,7]
instance = [49,'No','Travel_Frequently',279,'Research & Development',8,1,'Life Sciences',1,2,3,'Male',61,2,2,'Research Scientist',2,'Married',5130,24907,1,'Y','No',23,4,4,80,1,10,3,3,10,7,1,7]
data = pd.concat([data, pd.DataFrame({str(att): i for att, i in zip(data, instance)}, index=[0])])
data = data.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)

for i in cat:
    data[i] = (data[i].astype('category').cat.codes).apply(np.int64)

with open('Models.pickle', 'rb') as handle:
    models = pickle.load(handle)

max_unique_val = models['max'] + 1
models.pop('max')

y = data['Attrition']
X = data.drop(['Attrition'], axis=1)

kbins = KBinsDiscretizer(n_bins=max_unique_val, encode='ordinal', strategy='uniform')
for att in X:
    column = np.array(X[att])
    if len(np.unique(column)) > max_unique_val:
        column = column.reshape((len(column), 1))
        X[att] = kbins.fit_transform(column)

instance = data.iloc[-1]
instance = np.array(instance.drop('Attrition'))

scaler = StandardScaler()
X = scaler.fit_transform(X)
st.title('Predictions')
model = st.selectbox('Choose a model', models)
with st.expander('Existing instances'):
    i = st.number_input('Instance', 0, len(X))
    a = [i for i in attributes if i not in ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']]
    st.dataframe(pd.DataFrame({att: i for att, i in zip(a, list(data_og.iloc[i]))}, index=[0]))
    try:
        st.write('Predicted: ', models[model].predict(X[i])) 
        st.write('Ground truth:', y[i] if type(y[i]) is np.int64 else list(y[i])[0])
    except:
        data = np.array(X[i]).reshape(1, -1)
        st.write('Predicted: ', models[model].predict(data)[0])
        st.write('Ground truth:', y[i] if type(y[i]) is np.int64 else list(y[i])[0])

with st.expander('Unseen instances'):
    instance_txt = {att: i for att, i in zip(attributes, instance_txt) if att not in ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']}
    instance = st.text_input('Instance', ', '.join([str(i) for i in list(instance_txt.values())]))
    instance = [i if not i.isdigit() else int(i) for i in instance.split(', ')]
    
    instance = {att: i for att, i in zip(list(instance_txt.keys()), instance) if att not in ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']}
    st.dataframe(pd.DataFrame(instance, index=[0]))
    instance = np.array(data[-1])
    #for i in np.array(cat)[:-1]:
    #    temp = pd.DataFrame(instance, index=[0])
    #    instance[i] = int((temp[i].astype('category').cat.codes).apply(np.int64)[0])
    try:
        #instance = np.array(list(instance.values()))
        st.write('Predicted: ', models[model].predict(instance)[0])
    except:
        instance = instance.reshape(1, -1)
        st.write('Predicted: ', models[model].predict(instance)[0])
