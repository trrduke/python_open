
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

dir='D:/git/github/python_open/kaggle/Titanic/'

train_data=pd.read_csv(dir+'train.csv')
test_data=pd.read_csv(dir+'test.csv')
gs_data=pd.read_csv(dir+'gender_submission.csv')

full_data=train_data.append(test_data,ignore_index=True,sort=False)

full_data.info()

#数据预处理
def get_ticket_info(x):
    if ' ' not in x and 'LINE' not in x:
        a='Number'
    else:
        a=x.split(' ')[0].split('/')[0].split('.')[0].strip()
    return a

full_data['TicketInfo']=full_data['Ticket'].apply(get_ticket_info)
full_data['Fare']=full_data['Fare'].fillna(full_data['Fare'].mean())
full_data['Age']=full_data['Age'].fillna(full_data['Age'].mean())
full_data['Embarked']=full_data['Embarked'].fillna('S')
full_data['Cabin']=full_data['Cabin'].fillna('U')
full_data['Cabin']=full_data['Cabin'].map(lambda x:x[0])

def get_title(x):
    a=x.split(',')[1]
    b=a.split('.')[0].strip()
    return b

full_data['Title']=full_data['Name'].apply(get_title)

title_mapDict={
    'Capt':'Officer',
    'Col':'Officer',
    'Major':'Officer',
    'Jonkheer':'Royalty',
    'Don':'Royalty',
    'Sir':'Royalty',
    'Dr':'Officer',
    'Rev':'Officer',
    'the Countess':'Royalty',
    'Dona':'Royalty',
    'Mme':'Mrs',
    'Mlle':'Miss',
    'Ms':'Mrs',
    'Mrs':'Mrs',
    'Mr':'Mr',
    'Miss':'Mrs',
    'Master':'Master',
    'Lady':'Royalty'    
}

full_data['Title']=full_data['Title'].map(title_mapDict)

full_data['FamilySize']=full_data['SibSp']+full_data['Parch']+1

full_data.info()

oh_enc = full_data.loc[:, ['Sex', 'Cabin', 'Embarked', 'TicketInfo','Title']]
enc = OneHotEncoder(categories='auto')
enc.fit(oh_enc)
result = enc.transform(oh_enc).toarray()

ok_data = pd.concat([full_data, pd.DataFrame(result, columns=enc.get_feature_names())], axis=1)
ok_data.drop(['Sex', 'Cabin', 'Embarked', 'TicketInfo','Title','Name','Ticket'], axis=1, inplace=True)

ok_data=ok_data.set_index('PassengerId')

ok_columns=list(ok_data.columns)

ok_columns.remove('Survived')

X_data=ok_data.drop(['Survived'],axis=1)
Y_data=ok_data['Survived']

# X_data.describe([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]).T

X_data=StandardScaler().fit_transform(X_data)

X_data=pd.DataFrame(X_data,columns=ok_columns,index=ok_data.index)

ok_data=X_data.join(Y_data)

train_d=ok_data[~ok_data['Survived'].isnull()]
test_d=ok_data[ok_data['Survived'].isnull()]

PassengerId = test_d.index

x_test=test_d.drop(['Survived'],inplace=False,axis=1)

x_data=train_d.drop(['Survived'],inplace=False,axis=1)
y_data=train_d['Survived']

train_data_x=x_data
test_data_x=test_d.drop('Survived',axis=1)
train_data_y=y_data

'''
# 调参过程

scorel = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1,
                                 n_jobs=-1,
                                 random_state=123)
    score = cross_val_score(rfc, x_data, y_data,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,200,10),scorel)
plt.show()

scorel = []
for i in range(110,130):
    rfc = RandomForestClassifier(n_estimators=i+1,
                                 n_jobs=-1,
                                 random_state=123)
    score = cross_val_score(rfc, x_data, y_data,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel)))+110)
plt.figure(figsize=[20,5])
plt.plot(range(110,130),scorel)
plt.show()

scorel = []
for i in range(1,10):
    rfc = RandomForestClassifier(n_estimators=129,
                                 max_depth=i,
                                 n_jobs=-1,
                                 random_state=123)
    score = cross_val_score(rfc, x_data, y_data,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel)))+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,10),scorel)
plt.show()

# param_grid={'max_depth':np.arange(1,20,1)} #7是最优
# param_grid = {'min_samples_leaf':np.arange(1, 1+10, 1)} #1为最优
param_grid = {'max_features':np.arange(5,20,1)}
rfc = RandomForestClassifier(n_estimators=60 ,max_depth=7,min_samples_leaf=2,random_state=123  )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x_data, y_data)
GS.best_params_
'''

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import GradientBoostingClassifier as GBC

ada = ABC(n_estimators=48
          ,learning_rate=0.58)

rf=RandomForestClassifier(
    n_estimators =129
   ,max_depth=7
    ,min_samples_leaf=2
   ,max_features=6
  )

svm=SVC(kernel='rbf'
        ,gamma=0.024420530945486497
        ,C=0.6220408163265306
        ,cache_size=5000
       )

gb = GBC(n_estimators=145
         ,learning_rate=0.07
        ,random_state=150)

et = ETC(n_estimators=112)

knn = KNN(n_neighbors=6)

#交叉检验
scores = cross_val_score(knn, x_data, y_data, cv=10)  # for classification
print('预测准确度：',scores.mean())

'''
#调参
scorel = []
for i in range(0,10):
    clf = KNN(n_neighbors=i+1
              )
    score = cross_val_score(clf, x_data, y_data,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*1)+1)
plt.figure(figsize=[20,5])
plt.plot(range(0,10),scorel)
plt.show()

scorel = []
for i in range(0,10):
    clf = GBC(n_estimators=145
              ,learning_rate=0.07
              ,subsample=0.9+i/100
              ,random_state=150)
    score = cross_val_score(clf, x_data, y_data,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))/100+0.9))
plt.figure(figsize=[20,5])
plt.plot(range(0,10),scorel)
plt.show()
'''

#模型融合
from sklearn.model_selection import KFold
 
# Some useful parameters which will come in handy later on
ntrain = train_data_x.shape[0]
ntest = test_data_x.shape[0]
SEED = 0 #for reproducibility
NFOLDS = 7 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS,random_state=SEED,shuffle=False)

def get_out_fold(clf,x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i, (train_index,test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.fit(x_tr,y_tr)
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)
        
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

# Create Numpy arrays of train,test and target(Survived) dataframes to feed into our models
x_train = train_data_x.values   #Creates an array of the train data
x_test = test_data_x.values   #Creates an array of the test data
y_train = train_data_y.values

# Create our OOF train and test predictions.These base result will be used as new featurs
rf_oof_train,rf_oof_test = get_out_fold(rf,x_train,y_train,x_test)  # Random Forest
ada_oof_train,ada_oof_test = get_out_fold(ada,x_train,y_train,x_test)  # AdaBoost
et_oof_train,et_oof_test = get_out_fold(et,x_train,y_train,x_test)  # Extra Trees
gb_oof_train,gb_oof_test = get_out_fold(gb,x_train,y_train,x_test)  # Gradient Boost
knn_oof_train,knn_oof_test = get_out_fold(knn,x_train,y_train,x_test)  # KNeighbors
svm_oof_train,svm_oof_test = get_out_fold(svm,x_train,y_train,x_test)  # Support Vector
print("Training is complete")

x_train = np.concatenate((rf_oof_train,ada_oof_train,et_oof_train,gb_oof_train,knn_oof_train,svm_oof_train),axis=1)
x_test =np.concatenate((rf_oof_test,ada_oof_test,et_oof_test,gb_oof_test,knn_oof_test,svm_oof_test),axis=1)

from xgboost import XGBClassifier
 
gbm = XGBClassifier(n_estimators=200
                    ,max_depth=4
                    ,min_child_weight=2
                    ,gamma=0.9
                    ,subsample=0.8
                    ,colsample_bytree=0.8
                    ,objective='binary:logistic'
                    ,nthread=-1
                    ,scale_pos_weight=1)

#交叉检验
scores = cross_val_score(gbm, x_data, y_data, cv=10)  # for classification
print('预测准确度：',scores.mean())

scorel = []
for i in range(0,150,10):
    clf = XGBClassifier(n_neighbors=i+1
                        ,random_state=150
              )
    score = cross_val_score(clf, x_train, y_train,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1)
plt.figure(figsize=[20,5])
plt.plot(range(0,150,10),scorel)
plt.show()

gbm.fit(x_train,y_train)
predictions = gbm.predict(x_test).astype(int)

StackingSubmission = pd.DataFrame({'PassengerId':PassengerId,'Survived':predictions})
StackingSubmission.to_csv(dir+'titanic_pred_ensemble_1025.csv',index=False)
