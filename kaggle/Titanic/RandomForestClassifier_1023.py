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

# full_data.info()

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

rfc=RandomForestClassifier(
    n_estimators =129
   ,max_depth=7
    ,min_samples_leaf=2
   ,max_features=6
   ,random_state=123
                          )


#交叉检验
scores = cross_val_score(rfc, x_data, y_data, cv=10)  # for classification
print('预测准确度：',scores.mean())

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

rfc.fit(x_data, y_data)

pred_y=rfc.predict(x_test)

pred_y=pred_y.astype(int)

predf=pd.DataFrame({'PassengerId':x_test.index,'Survived':pred_y})

predf.to_csv(dir+'titanic_pred1023.csv',index=False)