import pickle
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import streamlit as st
import xgboost as xgb
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

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

st.title('IBM attrition estimator')
cat = ['OverTime',
'MaritalStatus',
'JobRole',
'Gender',
'EducationField',
'Department',
'BusinessTravel',
'Attrition']

models = {}
rocs = {}

data = pd.read_csv('datset.csv') 
#data["Attrition"] = data['Attrition'].astype('category').cat.codes
data = data.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)
for i in cat:
    data[i] = (data[i].astype('category').cat.codes).apply(np.int64)

max_unique_val = st.sidebar.number_input('Max unique values par attribute', 5, 1500)

kbins = KBinsDiscretizer(n_bins=max_unique_val, encode='ordinal', strategy='uniform')
for att in data:
    column = np.array(data[att])
    if len(np.unique(column)) > max_unique_val:
        column = column.reshape((len(column), 1))
        data[att] = kbins.fit_transform(column)

st.sidebar.markdown('Decision Tree')
max_depth_dt = st.sidebar.number_input('Max depth', 1, 50, key=0) 

st.sidebar.markdown('Random Forest')
num_trees = st.sidebar.number_input('Number of trees', 1, 50) 
min_samples_split = 2
max_depth_rf = st.sidebar.number_input('Max depth', 1, 20, key=1) 

y = data['Attrition']
X = data.drop(['Attrition'], axis=1)

col1, col2 = st.sidebar.columns(2)
run = col1.button('Run')
yes_tsne = col2.checkbox('TSNE')
if run:
    ##############################################################################
    if yes_tsne:
        start = time.time()
        tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=2000, learning_rate='auto').fit_transform(X)
        stop = time.time()
        with st.expander('T-SNE visualization of data disturibution'):
            st.write("T-SNE fitting time", stop-start, 'seconds')
            fig, ax = plt.subplots()
            ax.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=y[:], cmap='Spectral')
            col, _ = st.columns(2)
            col.pyplot(fig)
    ##############################################################################
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y), test_size=0.2, random_state=42)
    ##############################################################################
    start1 = time.time()
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    stop1 = time.time()
    start2 = time.time()
    ypred = clf.predict(X_test)
    stop2 = time.time()
    with st.expander('Naive Bayes classifier'):
        col1, col2 = st.columns([2, 3])
        cm = confusion_matrix(y_test, ypred)
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, fmt='g')
        col1.pyplot(fig)
        col2.text('metric' + classification_report(y_test, ypred, digits=4)[6:]) 
        st.write("Training time", stop1-start1, 'seconds')
        st.write("Prediction time", stop2-start2, 'seconds') 
    models['Naive Bayes classifier'] = clf
    FPR, TPR, _ = roc_curve(y_test, ypred)
    rocs['Naive Bayes classifier'] = (FPR, TPR)
    ##############################################################################
    start1 = time.time()
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    stop1 = time.time()
    start2 = time.time()
    ypred = clf.predict(X_test)
    stop2 = time.time()
    with st.expander('KNN with k=3'):
        col1, col2 = st.columns([2, 3])
        cm = confusion_matrix(y_test, ypred)
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, fmt='g')
        col1.pyplot(fig)
        col2.text('metric' + classification_report(y_test, ypred, digits=4)[6:])
        st.write("Training time", stop1-start1, 'seconds')
        st.write("Prediction time", stop2-start2, 'seconds')
    models['KNN'] = clf
    FPR, TPR, _ = roc_curve(y_test, ypred)
    rocs['KNN'] = (FPR, TPR)
    ##############################################################################
    start1 = time.time()
    clf = DecisionTree(min_samples_split, max_depth_dt)
    clf.fit(X_train, y_train)
    stop1 = time.time()
    start2 = time.time()
    ypred = clf.predict(X_test)
    stop2 = time.time()
    with st.expander('Decision Tree'):
        col1, col2 = st.columns([2, 3])
        cm = confusion_matrix(y_test, ypred)
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, fmt='g')
        col1.pyplot(fig)
        col2.text('metric' + classification_report(y_test, ypred, digits=4)[6:])
        st.write("Training time", stop1-start1, 'seconds')
        st.write("Prediction time", stop2-start2, 'seconds')
    models['Decision Tree'] = clf
    FPR, TPR, _ = roc_curve(y_test, ypred)
    rocs['Decision Tree'] = (FPR, TPR)
    ##############################################################################
    start1 = time.time()
    clf = RandomForest(num_trees, min_samples_split, max_depth_rf)
    clf.fit(X_train, y_train)
    stop1 = time.time()
    start2 = time.time()
    ypred = clf.predict(X_test)
    stop2 = time.time()
    with st.expander('Random Forest'):
        col1, col2 = st.columns([2, 3])
        cm = confusion_matrix(y_test, ypred)
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, fmt='g')
        col1.pyplot(fig)
        col2.text('metric' + classification_report(y_test, ypred, digits=4)[6:])
        st.write("Training time", stop1-start1, 'seconds')
        st.write("Prediction time", stop2-start2, 'seconds')
    models['Random Forest'] = clf
    FPR, TPR, _ = roc_curve(y_test, ypred)
    rocs['Random Forest'] = (FPR, TPR)
    ##############################################################################
    start1 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        missing=-999,
        random_state=2019
    )
    clf.fit(X_train, y_train)
    stop1 = time.time()
    start2 = time.time()
    ypred = clf.predict(X_test)
    stop2 = time.time()
    with st.expander('XGBoost'):
        col1, col2 = st.columns([2, 3])
        cm = confusion_matrix(y_test, ypred)
        fig, ax = plt.subplots()
        sn.heatmap(cm, annot=True, fmt='g')
        col1.pyplot(fig)
        col2.text('metric' + classification_report(y_test, ypred, digits=4)[6:])
        st.write("Training time", stop1-start1, 'seconds')
        st.write("Prediction time", stop2-start2, 'seconds')
    models['XGBoost'] = clf
    FPR, TPR, _ = roc_curve(y_test, ypred)
    rocs['XGBoost'] = (FPR, TPR)
    ##############################################################################
    models['max'] = max_unique_val
    with open('Models.pickle', 'wb') as handle:
        pickle.dump(models, handle)
    fig, ax = plt.subplots()
    #svc_disp = RocCurveDisplay.from_estimator(models['XGBoost'], X_test, y_test)

    #roc = roc_curve(y_test, ypred, pos_label=2)
    #ax.plot(roc[0], roc[-1])
    #fpr, tpr, thresholds = metrics.roc_curve(y_test, ypred,)
    #roc_auc = metrics.auc(fpr, tpr)
    #display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                          estimator_name='example estimator')
    #display.plot()  
    
    for mod, roc in rocs.items():
        ax.plot(roc[0], roc[1], label=mod)
    ax.legend()
    ax.set_title('ROC curves')
    with st.expander('Roc curves'):
        col, _ = st.columns(2)
        col.pyplot(fig)
        
