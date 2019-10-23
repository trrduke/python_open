

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import datetime
from time import time


dir='D:/git/github/python_open/kaggle/Titanic/'

train_data=pd.read_csv(dir+'train.csv')
test_data=pd.read_csv(dir+'test.csv')
gs_data=pd.read_csv(dir+'gender_submission.csv')

full_data=train_data.append(test_data,ignore_index=True,sort=False)

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

x_test=test_d.drop(['Survived'],inplace=False,axis=1)

x_data=train_d.drop(['Survived'],inplace=False,axis=1)
y_data=train_d['Survived']

# xtrain,xtest,ytrain,ytest=train_test_split(x_data,y_data,test_size=0.3,random_state=123)

'''
#对比集中Kernel的准确率
Kernel=['linear'
        ,'poly'
        ,'rbf'
        ,'sigmoid']
for kernel in Kernel:
    time0=time()
    clf=SVC(kernel=kernel
            ,gamma="auto"
            ,cache_size=4000
            ,degree=1
            )
    scores = cross_val_score(clf, x_data, y_data, cv=10)  # for classification
    print("The accuracy under kernal %s is %f" % (kernel,scores.mean()))
    print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
'''

'''
#gamma参数调整，学习曲线
score=[]
gamma_range=np.logspace(-10,1,50)
for i in gamma_range:
    clf=SVC(kernel='rbf',gamma=i,cache_size=4000)
    scores = cross_val_score(clf, x_data, y_data, cv=10)
    score.append(scores.mean())
print(max(score),gamma_range[score.index(max(score))])
plt.plot(gamma_range,score)
plt.show()

score=[]
gamma_range=np.logspace(-4,-1.5,50)

for i in gamma_range:
    clf=SVC(kernel='rbf',gamma=i,cache_size=4000)
    scores = cross_val_score(clf, x_data, y_data, cv=10)
    score.append(scores.mean())
print(max(score),gamma_range[score.index(max(score))])
plt.plot(gamma_range,score)
plt.show()
'''

'''
poly参数网格搜索
time0=time()
gamma_range=np.logspace(-3,-1,20)
coef0_range=np.linspace(0,5,10)
param_grid = dict(gamma=gamma_range
                  ,coef0=coef0_range)
cv=StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=123)
GS = GridSearchCV(
    SVC(kernel='poly'
        ,degree=1
        ,cache_size=4000)
        ,param_grid=param_grid,cv=cv)
GS.fit(x_data, y_data)
print('The best parameters are %s with a score of %0.5f'
      % (GS.best_params_,GS.best_score_))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
'''

'''
#C参数调整，学习曲线
score=[]
c_range=np.linspace(0.01,30,50)
for i in c_range:
    clf=SVC(kernel='rbf',C=i,gamma=0.024420530945486497,cache_size=5000)
    scores = cross_val_score(clf, x_data, y_data, cv=10)
    score.append(scores.mean())
print(max(score),c_range[score.index(max(score))])
plt.plot(c_range,score)
plt.show()
'''

#根据最优参数生成模型
clf=SVC(kernel='rbf'
        ,gamma=0.024420530945486497
        ,C=0.6220408163265306
        ,cache_size=5000
       )

'''
#交叉检验
scores = cross_val_score(clf, x_data, y_data, cv=10)  # for classification
print('预测准确度：',scores.mean())
'''

clf.fit(x_data,y_data)

pred_y=clf.predict(x_test)
pred_y=pred_y.astype(int)
predf=pd.DataFrame({'PassengerId':x_test.index,'Survived':pred_y})
predf.to_csv(dir+'titanic_pred_svc_1023.csv',index=False)