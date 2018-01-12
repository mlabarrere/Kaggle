
import pandas as pd
from src.doPreprocess import preprocess
from src.runGradient import run_GBR

join_raw = pd.read_csv("Data/joined_data.csv",
                       error_bad_lines=False, 
                       index_col=False, 
                       dtype='unicode', 
                       keep_default_na=False)


X_train, Y_train, X_test, Y_test = preprocess(join_raw)

run_GBR(X_train, Y_train, X_test, Y_test)