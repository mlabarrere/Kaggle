import gc
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

join_raw = pd.read_csv("Data/joined_data.csv",
                       error_bad_lines=False, 
                       index_col=False, 
                       dtype='unicode', 
                       keep_default_na=False)



from sklearn.linear_model import Lasso


# In[25]:

lss_rg=Lasso(alpha=1.0,
             fit_intercept=True,
             normalize=False,
             precompute=False,
             copy_X=True,
             max_iter=10000000,
             tol=0.0001,
             warm_start=True,
             positive=False,
             random_state=RANDOM_SEED,
             selection='cyclic'
             )

lss_rg.fit(X_train, Y_train)
y_pred_lss = lss_rg.predict(X_test)
mean_absolute_error(Y_test, y_pred_lss)
coeff_lasso=pd.DataFrame(lss_rg.coef_*10000000000000,index=X_train.columns.values)


non_linear_combinaison=['bathroomcnt',
 'bedroomcnt',
 'calculatedbathnbr',
 'finishedsquarefeet12',
 'fips',
 'propertylandusetypeid',
 'regionidcounty',
 'regionidzip',
 'roomcnt',
 'yearbuilt',
 'assessmentyear',
 'landtaxvaluedollarcnt']


X_train_light=X_train.drop(non_linear_combinaison,axis=1,inplace=False)
X_test_light=X_test.drop(non_linear_combinaison,axis=1,inplace=False)


X_train=X_train_light
X_test=X_test_light



del lss_rg
del y_pred_lss
del coeff_lasso
del non_linear_combinaison
del X_train_light
del X_test_light
gc.collect()


from sklearn.linear_model import ElasticNet


# In[35]:

en_rg = ElasticNet(alpha=1.0,
           l1_ratio=0.5, 
           fit_intercept=True, 
           normalize=False, 
           precompute=False, 
           max_iter=1000000, 
           copy_X=True, 
           tol=0.0001, 
           warm_start=True, 
           positive=False, 
           random_state=RANDOM_SEED, 
           selection='cyclic'
          )


# In[36]:

from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:

gb_rg = GradientBoostingRegressor(loss='huber', 
                                  learning_rate=0.1, 
                                  n_estimators=100, 
                                  subsample=1.0, 
                                  criterion='mae', 
                                  min_samples_split=2, 
                                  min_samples_leaf=1, 
                                  min_weight_fraction_leaf=0.0, 
                                  max_depth=3, 
                                  min_impurity_split=1e-07, 
                                  init=None, 
                                  random_state=RANDOM_SEED, 
                                  max_features=None, 
                                  alpha=0.9, 
                                  verbose=0, 
                                  max_leaf_nodes=None, 
                                  warm_start=False, 
                                  presort='auto'
                                 )


# In[ ]:

gb_rg.fit(X_train, Y_train)


# In[ ]:

y_pred_gb = gb_rg.predict(X_test)


# In[ ]:

mean_absolute_error(Y_test, y_pred_gb)

