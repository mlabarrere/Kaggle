from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
RANDOM_SEED = 42


def run_GBR(X_train, Y_train, X_test, Y_test):
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

  gb_rg.fit(X_train, Y_train)
  y_pred_gb = gb_rg.predict(X_test)
  print(mean_absolute_error(Y_test, y_pred_gb))
