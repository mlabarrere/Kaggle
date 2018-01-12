from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

RANDOM_SEED = 42

def fitRandomForest(X_train, Y_train, X_test, Y_test):
    rf_rg = RandomForestRegressor(n_estimators=10,
                                  criterion='mae',
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features='auto',
                                  max_leaf_nodes=None,
                                  min_impurity_split=1e-07,
                                  bootstrap=True,
                                  oob_score=False,
                                  n_jobs=2,
                                  random_state=RANDOM_SEED,
                                  verbose=0,
                                  warm_start=False
                                  )
    rf_rg.fit(X_train, Y_train)
    y_pred_rf = rf_rg.predict(X_test)
    return mean_absolute_error(Y_test, y_pred_rf)

