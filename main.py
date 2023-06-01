import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

#iCorrect

warnings.filterwarnings("ignore", category=ConvergenceWarning)      #ignore unnecessary warnings

data = pd.read_csv(r'data.csv')                                     #dataset path

new_column_names = {                                                #rename column names with short values to get rid of long questions 
    'Yaşınız' : 'age' ,
    'Cinsiyyetiniz' : 'gender' ,
    'Şu anki stres durumunuz nedir ?' : 'stress_level' ,
    'Aile evinde mi yaşıyorsunuz ?' : 'residency_status' ,
    'Ailenizde sigara içen biri(leri) var mı?' : 'family_smokers' ,
    'Alkol kullanıyor musunuz?' : 'alcohol_use' ,
    'Haftada kaç kez sosyalleşirsiniz / dışarı çıkıyorsunuz?' : 'social_days' ,
    'Yakın arkadaş çevrenizin çoğunluğu sigara içiyor mu?' : 'friend_smokers' ,
    'Herhangi bir akciğer hastalığınız var mı?' : 'lung_disease' ,
    'Ergenlik döneminde hiç sigara içtiniz mi?' : 'teenage_smoking' ,
    'Şu an aktif olarak sigara kullanıyor musunuz?' : 'smoking_status' ,
}
data = data.rename(columns=new_column_names)



                                                         #define data types
data['age'] = data['age'].astype('int')
data['gender'] = data['gender'].replace('Erkek',0 ).replace('Kadın',1 )
data['stress_level'] = data['stress_level'].astype('int')
data['residency_status'] = data['residency_status'].replace('Evet',True).replace('Hayır',False)
data['family_smokers'] = data['family_smokers'].replace('Evet',True).replace('Hayır',False)
data['alcohol_use'] = data['alcohol_use'].replace('Evet',True).replace('Hayır',False)
data['social_days'] = data['social_days'].replace('0-2',1).replace('3-5',2).replace('6-7',3)
data['friend_smokers'] = data['friend_smokers'].replace('Evet',True).replace('Hayır',False)
data['lung_disease'] = data['lung_disease'].replace('Evet',True).replace('Hayır',False)
data['teenage_smoking'] = data['teenage_smoking'].replace('Evet',True).replace('Hayır',False)
data['smoking_status'] = data['smoking_status'].replace('Evet',True).replace('Hayır',False)


FEATURES = [                                                        #define our future and target parameters
    'age',
    'gender',
    'stress_level',
    'residency_status',
    'family_smokers',
    'alcohol_use',
    'social_days',
    'friend_smokers',
    'lung_disease',
    'teenage_smoking',
]  

TARGET = 'smoking_status'

X = data[FEATURES]
y = data[TARGET]

data  = preprocessing.normalize(data)                               #data normalization

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)   # Train and Test


logreg = LogisticRegression()                                      #classifying with logistic regression 
logreg.fit(x_train, y_train)
logreg_pred= logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Logistic Regression test accuracy: {:.2f}%".format(logreg_acc*100))            #printing test accuracy and confusion matrix 
print( "\n" )
print(classification_report(y_test, logreg_pred)) 
print( "\n" )    

SVCmodel = LinearSVC()                                             #classifying with linear svc
SVCmodel.fit(x_train, y_train)
svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print ("SVC test accuracy: {:.2f}%".format(svc_acc*100))
print( "\n" )
print(classification_report(y_test, svc_pred))
print( "\n" )

RFCmodel = RandomForestClassifier()                                #classifying with random forest
RFCmodel.fit(x_train,y_train)
rfc_pred = RFCmodel.predict(x_test)
rfc_acc = accuracy_score(rfc_pred, y_test)
print ("Random forest test accuracy: {:.2f}%".format(rfc_acc*100))
print( "\n" )
print(classification_report(y_test, rfc_pred))
print( "\n" )

BernNBmodel = BernoulliNB()                                        #classifying with bernoulli naive bayes
BernNBmodel.fit(x_train,y_train)
bern_predict = BernNBmodel.predict(x_test)
bern_acc = accuracy_score(bern_predict,y_test) 
print ("Bernoulli Naive Bayes test accuracy: {:.2f}%".format(bern_acc*100))
print( "\n" )
print(classification_report(y_test, bern_predict))
print( "\n" )

SGDCmodel = SGDClassifier()                                      #classifying with stochastic gradient descent
SGDCmodel.fit(x_train,y_train)
sgdc_pred = SGDCmodel.predict(x_test)
sgdc_acc = accuracy_score(sgdc_pred,y_test)
print ("Stochastic Gradient Descent test accuracy: {:.2f}%".format(sgdc_acc*100))
print( "\n" )
print(classification_report(y_test, sgdc_pred))
print( "\n" )


#Selecting 7 best features and applying PCA with this features

logreg = make_pipeline(PCA(n_components=7), LogisticRegression())
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Logistic Regression (with PCA) test accuracy: {:.2f}%".format(logreg_acc * 100))
print("\n")
print(classification_report(y_test, logreg_pred))
print("\n")

SVCmodel = make_pipeline(PCA(n_components=7), LinearSVC())
SVCmodel.fit(x_train, y_train)
svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("Linear SVC (with PCA) test accuracy: {:.2f}%".format(svc_acc * 100))
print("\n")
print(classification_report(y_test, svc_pred))
print("\n")

RFCmodel = make_pipeline(PCA(n_components=7), RandomForestClassifier())
RFCmodel.fit(x_train, y_train)
rfc_pred = RFCmodel.predict(x_test)
rfc_acc = accuracy_score(rfc_pred, y_test)
print("Random Forest (with PCA) test accuracy: {:.2f}%".format(rfc_acc * 100))
print("\n")
print(classification_report(y_test, rfc_pred))
print("\n")

BernNBmodel = make_pipeline(PCA(n_components=7), BernoulliNB())
BernNBmodel.fit(x_train, y_train)
bern_predict = BernNBmodel.predict(x_test)
bern_acc = accuracy_score(bern_predict, y_test)
print("Bernoulli Naive Bayes (with PCA) test accuracy: {:.2f}%".format(bern_acc * 100))
print("\n")
print(classification_report(y_test, bern_predict))
print("\n")

SGDCmodel = make_pipeline(PCA(n_components=7), SGDClassifier())
SGDCmodel.fit(x_train, y_train)
sgdc_pred = SGDCmodel.predict(x_test)
sgdc_acc = accuracy_score(sgdc_pred, y_test)
print("Stochastic Gradient Descent (with PCA) test accuracy: {:.2f}%".format(sgdc_acc * 100))
print("\n")
print(classification_report(y_test, sgdc_pred))
print("\n")



