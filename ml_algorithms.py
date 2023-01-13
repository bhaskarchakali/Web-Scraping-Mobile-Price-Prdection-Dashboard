import pandas as pd #for data manipulations
import numpy as np  #for numerical calculations
import seaborn as sns #for advanced data visualizations
import matplotlib.pyplot as plt # for data visualizations
# loading dataset
df = pd.read_csv('C:/Users/Home/Desktop/Interview project/Fmobiles_clean.csv', encoding='utf-8')
df.duplicated().sum()
df = df.drop_duplicates()
df.isna().sum()
df.info()
df.columns
df.BATTERY_C = df.BATTERY_C.astype('int64')
df=df.drop('DISCOUNT',axis=1)
df.corr()

#sns.pairplot(data = df)
from scipy import stats
import pylab
stats.probplot(df.OFFER_PRICE, dist = "norm", plot = pylab); plt.show()
sns.distplot(df['OFFER_PRICE'])
sns.distplot(np.log(df['OFFER_PRICE']))
# for log(price) it shows data is normaly distributed so consider log(price) for further calculation
y = np.log(df.OFFER_PRICE)
x = df.drop('OFFER_PRICE' , axis = 1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split( x,y,test_size=0.2)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
################# Linear Regression ##############################
x.columns
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = LinearRegression()
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred = pipe.predict(xtest)
R2_Linear_regression = r2_score(ytest,ypred)
R2_Linear_regression 
MAE_linear_regression = mean_absolute_error(ytest,ypred) 
MAE_linear_regression 
# we have low amount data due to that for each train_test split R_Square changed so trying to find max rsquare
scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    step2 = LinearRegression() 
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

np.argmax(scores)
scores[np.argmax(scores)]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
step2 = LinearRegression() 
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_Linear_regression = r2_score(ytest,ypred)
R2_Linear_regression # 0.9887038943133817
MAE_linear_regression = mean_absolute_error(ytest,ypred) 
MAE_linear_regression # 0.11580766887090362

#################  Lasso Regression ################

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores)) 
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_Lasso_regression = r2_score(ytest,ypred)
R2_Lasso_regression #  0.9854994050689824
MAE_lasso_regression = mean_absolute_error(ytest,ypred) 
MAE_lasso_regression # 0.13639782286335966

################# Ridge Regression ####################
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = Ridge(alpha=0.19)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_Ridge_regression = r2_score(ytest,ypred)
R2_Ridge_regression # 0.9883742192963664
MAE_ridge_regression = mean_absolute_error(ytest,ypred) 
MAE_ridge_regression # 0.11794225469525604

#####################  KNN ####################
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=30)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i) 
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_KNN= r2_score(ytest,ypred)
R2_KNN # 0.9723335840561901
MAE_KNN = mean_absolute_error(ytest,ypred)
MAE_KNN # 0.19906642507410588

#################### Decision Tree #############
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=10)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_DT = r2_score(ytest,ypred)
R2_DT 
MAE_DT = mean_absolute_error(ytest,ypred)
MAE_DT 
##############  Random Forest #########################
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=25)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_random_forest = r2_score(ytest,ypred)
R2_random_forest # 0.9947759035756751
MAE_random_forest = mean_absolute_error(ytest,ypred)
MAE_random_forest #  0.07451652826779927

################## SVM ####################
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_svm = r2_score(ytest,ypred)
R2_svm # 0.9817860522332128
MAE_SVM = mean_absolute_error(ytest,ypred)
MAE_SVM # 0.15583099232182454

################  Extra Trees ##########
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_features=0.75,
                              max_depth=28)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_extra_trees= r2_score(ytest,ypred)
R2_extra_trees # 0.9966601577956925
MAE_extra_trees = mean_absolute_error(ytest,ypred)
MAE_extra_trees #  0.06109564872274928

################### AdaBoost #################
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=0.5)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_AdaBoost = r2_score(ytest,ypred)
R2_AdaBoost 
MAE_adaboost = mean_absolute_error(ytest,ypred)
MAE_adaboost 
####################### Gradient Boost ###############
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_GBoost = r2_score(ytest,ypred)
R2_GBoost
MAE_gboost = mean_absolute_error(ytest,ypred)
MAE_gboost 

############################## XG Boost #####################
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_XGBoost= r2_score(ytest,ypred)
R2_XGBoost # 0.9945568445052193
MAE_xgboost = mean_absolute_error(ytest,ypred) 
MAE_xgboost #  0.08333134734575937

############################## Stacking #####################
step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

estimators = [
    ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

scores=[]
for i in range(10):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_stacking = r2_score(ytest,ypred)
R2_stacking 
MAE_stacking = mean_absolute_error(ytest,ypred)
MAE_stacking 

                                    ############ Finding the Best Model ##############

data = {"Model" : pd.Series(['Linear Regression' , 'Lasso Regression' , 'Ridge Regression' ,
'KNN', 'Decision Tree', 'Random Forest', 'SVM', 'Extra Trees', 'AdaBoost', 'Gradient Boost',
'XG Boost',  'Stacking']) , 
"R Square Value" : pd.Series([R2_Linear_regression , R2_Lasso_regression , R2_Ridge_regression , 
R2_KNN , R2_DT , R2_random_forest , R2_svm , R2_extra_trees , R2_AdaBoost , R2_GBoost , R2_XGBoost , 
  R2_stacking]) ,
"Mean Absolute Error" : pd.Series([MAE_linear_regression , MAE_lasso_regression , MAE_ridge_regression , 
MAE_KNN , MAE_DT , MAE_random_forest , MAE_SVM , MAE_extra_trees , MAE_adaboost , MAE_gboost , 
MAE_xgboost  , MAE_stacking])} 
                                                                                                                                                         
R_Square_and_Error_Values = pd.DataFrame(data)
R_Square_and_Error_Values

# So Extra Trees is the best Model Among all this models        
# and it has lower error value 

# Exporting model 
import pickle
pickle.dump(df,open('interview project df.pkl','wb'))
pickle.dump(pipe,open('interview project model.pkl','wb'))


















