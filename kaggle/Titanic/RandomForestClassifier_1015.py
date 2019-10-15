
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


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

'''
#用OneHotEncoder进行onehot编码
oh_enc = full_data.loc[:, ['Pclass', 'vacation_days', 'day_after_vacation', 'special_day']]

enc = OneHotEncoder(categories='auto')
enc.fit(oh_enc)
result = enc.transform(oh_enc).toarray()

date_info = pd.concat([date_info, pd.DataFrame(result, columns=enc.get_feature_names())], axis=1)
date_info.drop(['week', 'vacation_days', 'day_after_vacation', 'special_day'], axis=1, inplace=True)
'''

oh_enc = full_data.loc[:, ['Sex', 'Cabin', 'Embarked', 'TicketInfo','Title']]
enc = OneHotEncoder(categories='auto')
enc.fit(oh_enc)
result = enc.transform(oh_enc).toarray()

ok_data = pd.concat([full_data, pd.DataFrame(result, columns=enc.get_feature_names())], axis=1)
ok_data.drop(['Sex', 'Cabin', 'Embarked', 'TicketInfo','Title','Name','Ticket'], axis=1, inplace=True)

coorDf=ok_data.corr()
coorDf['Survived'].sort_values(ascending=False)

train_d=ok_data[~ok_data['Survived'].isnull()]

test_d=ok_data[ok_data['Survived'].isnull()]

x_test=test_d.drop(['Survived'],inplace=False,axis=1)

x_data=train_d.drop(['Survived'],inplace=False,axis=1)
y_data=train_d['Survived']

rfc=RandomForestClassifier(n_estimators =60
                           ,max_depth=5
                           ,max_features=11
                           ,random_state=123
                          )

#交叉检验
scores = cross_val_score(rfc, x_data, y_data, cv=10)  # for classification
print('预测准确度：',scores.mean())

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
for i in range(1,10):
    rfc = RandomForestClassifier(n_estimators=60,
                                 max_depth=i,
                                 n_jobs=-1,
                                 random_state=123)
    score = cross_val_score(rfc, x_data, y_data,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel)))+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,10),scorel)
plt.show()

# param_grid={'max_depth':np.arange(1,20,1)} #5是最优
# param_grid = {'min_samples_leaf':np.arange(1, 1+10, 1)} #1为最优
param_grid = {'max_features':np.arange(10,20,1)}
rfc = RandomForestClassifier(n_estimators=60 ,max_depth=5,random_state=123  )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x_data, y_data)
GS.best_params_

rfc.fit(x_data, y_data)

pred_y=rfc.predict(x_test)


gs_data.info()

pred_y=pred_y.astype(int)

PassengerId=x_test['PassengerId']

predf=pd.DataFrame({'PassengerId':PassengerId,'Survived':pred_y})

predf.to_csv(dir+'titanic_pred.csv',index=False)