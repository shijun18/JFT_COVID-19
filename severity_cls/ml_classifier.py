import os 
import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

params_dict = {
  'svm':{
    'C':[0.001,0.01,0.1,1,10],
    'gamma':[0.001,0.01,0.1,1,10]
  },
  'decision tree':{
    'max_depth':range(3,15),
  },
  'random forest':{
    'n_estimators':range(10,100,5),
    'criterion':['gini','entropy']
  },
  'mlp':{
    'alpha':[0.001,0.01,0.1,1,10],
    'hidden_layer_sizes':[(5,),(10,),(15,),(20,),(10,10),(7,7,7),(5,5,5,5)],
    'solver':['lbfgs'],
    'activation':['logistic'],
    'learning_rate':['constant','invscaling']
  },
}


METRICS = {
  'Accuracy':'accuracy',
  'Recall':'recall',
  'Precision':'precision',
  'F1':'f1',
  'AUC':'roc_auc'
  }

class ML_Classifier(object):
  '''
  Machine Learning Classifier for the classification
  Args:
  - clf_name, string, __all__ = ['lasso','knn','svm','decision tree','random forest','extra trees','bagging','mlp','xgboost']
  - params_dict, dict, parameters setting of the specified classifer
  '''
  def __init__(self,clf_name=None,params=None): 
    super(ML_Classifier,self).__init__()  
    self.clf_name = clf_name
    self.params = params
    self.clf = self._get_clf()  
    np.random.seed(0)
  

  def trainer(self,df,target_key,test_size=0.2,random_state=21,metric=None,k_fold=5,scaler_flag=False):
    params = self.params
    data_x,target_y = self.extract_df(df,target_key) 
    
    x_train,x_test,y_train,y_test = train_test_split(data_x,target_y,
                                            random_state=random_state,
                                            test_size=test_size,
                                            shuffle=True)
    if scaler_flag:
      scaler = StandardScaler()
      x_train = scaler.fit_transform(x_train)
      x_test= scaler.fit_transform(x_test)
    
    kfold = KFold(n_splits=k_fold)
    grid = GridSearchCV(estimator=self.clf,
                        param_grid=params,
                        cv=kfold,
                        scoring=metric,
                        refit='Recall',
                        return_train_score=True)
    grid = grid.fit(x_train,y_train)

    best_score = grid.best_score_
    best_model = grid.best_estimator_
    test_score = best_model.score(x_test,y_test)
    print("Recall Evaluation:")
    print("Best score:{}".format(best_score))
    print("Test score:{}".format(test_score))
    
    print('Best parameters:')
    for key in params.keys():
      print('%s:'%key)
      print(best_model.get_params()[key])
    
    if self.clf_name == 'random forest'
      new_grid = RandomForestClassifier(random_state=0,bootstrap=True)
      new_grid.set_params(**grid.best_params_)

      new_grid = new_grid.fit(x_train,y_train) 
      importances = new_grid.feature_importances_
      feat_labels = df.columns
      # print(feat_labels)
      indices = np.argsort(importances)[::-1]

      # print(indices)
      for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1,30, feat_labels[indices[f]], importances[indices[f]]))


    classifier = self._get_clf() 
    classifier.set_params(**grid.best_params_)
    classifier = classifier.fit(x_train,y_train) 
    test_result = classifier.predict(x_test)
    print(classification_report(y_test, test_result, target_names=['no_ill','ill'],output_dict=True))
    test_prob = classifier.predict_proba(x_test)
    return best_model,test_result,test_prob,y_test

    
  def _get_clf(self):
    if self.clf_name == 'svm':
      classifer = SVC(kernel='rbf',random_state=0,probability=True)
    elif self.clf_name == 'decision tree':
      classifer = DecisionTreeClassifier(random_state=0)
    elif self.clf_name == 'random forest':
      classifer = RandomForestClassifier(random_state=0,bootstrap=True)
    elif self.clf_name == 'mlp':
      classifer = MLPClassifier(max_iter=2000,warm_start=True,random_state=0)
    
    return classifer  


  def extract_df(self,df,target_key):
    y = df[target_key]
    del df[target_key]
    x = df
    
    return x.values,y.values



