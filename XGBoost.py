import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
####################
        #params = {"eta": 0.05,"max_depth": 5,"min_child_weight": 4,"silent": 1, #"subsample": 0.7, #"colsample_bytree": 0.7,"seed": 1}

nTrees=400
print ('Loading data')

USE_VALID=True
if USE_VALID:
    tag='plusValid'
    df=pd.read_csv('../input/train_valid.csv')
else:
    tag=''
    df=pd.read_csv('../input/numerai_training_data.csv')
        
trainX=df.drop(['id','era','data_type','target'],axis=1)
trainY=df[['target']]
df=''
features=trainX.columns
target=['target']
	
print('Starting XGBOOST training: nTrees=',nTrees)
#params={"learning_rate":0.1,"objective": "binary:logistic","eta": 0.1,"max_depth": 4,"min_child_weight": 4,"silent": 1, "seed": 1}
#params={'learning_rate' :0.1,'n_estimators':nTrees,'max_depth':4,'min_child_weight':6,'gamma':0.2,'subsample':0.8,'colsample_bytree':0.8,'objective':'binary:logistic','scale_pos_weight':1,'random_state':27}
params={'learning_rate' :0.1,'n_estimators':nTrees,'max_depth':4,'min_child_weight':4,'gamma':0.0,'objective':'binary:logistic','scale_pos_weight':1,'random_state':27}
gbm = xgb.train(params,xgb.DMatrix(trainX, trainY), nTrees)
print('Finished XGBoost training')

print('Testing XGBoost')

test=pd.read_csv('../input/numerai_tournament_data.csv')
valid=test[test['data_type']=='validation']

print ('Do your scoring here')
validX=valid.drop(['id','era','data_type','target'],axis=1)
validY=valid['target']
valid_pred=[]
valid_pred+=list(gbm.predict(xgb.DMatrix(validX)))
score=log_loss(validY, valid_pred, eps=1e-15, normalize=True)
roc=roc_auc_score(validY, valid_pred)
print ('Done Scoring! LogLoss of ',score,'should submit: ', score<-np.log(0.5))
print ("ROC AUC score: ",roc)
	
testX=test.drop(['id','era','data_type','target'],axis=1)

testY_XGB=[]
testY_XGB+= list(gbm.predict(xgb.DMatrix(testX)))
print('Testing done.')				
 
print ('Writing Submission')
filename='XGB_Out'+tag+'_'+str(nTrees)
submission = pd.DataFrame({"id": test["id"], "probability": testY_XGB})
submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
print (submission.head())
submission.to_csv('../output/'+filename+'_'+str(score)[0:7]+'.csv', index=False)	
print ('Finished submission')


                   
