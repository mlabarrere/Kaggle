from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_SEED = 42


def convertorZillow(data_df):
    # Set everything possible as integer/float
    data_df = data_df.apply(pd.to_numeric, errors='ignore')
    # Set te date into date format
    data_df['transactiondate'] = pd.to_datetime(data_df['transactiondate'])
    # Three boolean variables
    data_df["hashottuborspa"]=data_df["hashottuborspa"].astype('bool')
    data_df["fireplaceflag"]=data_df["fireplaceflag"].astype('bool')
    data_df["taxdelinquencyflag"]=data_df["taxdelinquencyflag"].astype('bool')
    # Two string variables
    data_df["propertycountylandusecode"]=data_df["propertycountylandusecode"].astype(str)
    data_df["propertyzoningdesc"]=data_df["propertyzoningdesc"].astype(str)
    return data_df

def medianFiller(data_df):
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    return pd.DataFrame(imp.fit_transform(data_df), columns=data_df.columns.values)

def preprocess(join_raw):
    join_clean = convertorZillow(join_raw)
    join_clean = join_clean.select_dtypes(include=['int64', 'float64'])
    join_clean.dropna(axis=1, how='any', thresh=70000, subset=None, inplace=True)
    join_clean = medianFiller(join_clean)

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

    join_clean.drop(non_linear_combinaison, axis=1, inplace=True)             

    train_set, test_set = train_test_split(join_clean,
                                           test_size=0.2,
                                           random_state=RANDOM_SEED)

    logerror_avg = train_set['logerror'].mean()
    logerror_std = train_set['logerror'].std()

    train_set_outliers = train_set.where((train_set["logerror"] < logerror_avg - 2 * logerror_std) &
                                         (train_set["logerror"] > logerror_avg + 2 * logerror_std)
                                         )

    train_set = train_set.where((train_set["logerror"] >= logerror_avg - 2 * logerror_std) &
                                (train_set["logerror"] <= logerror_avg + 2 * logerror_std)
                                )

    train_set.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)

    Y_train = train_set['logerror']
    X_train = train_set.drop(["logerror", "parcelid"], axis=1, inplace=False)

    Y_test = test_set["logerror"]
    X_test = test_set.drop(["logerror", "parcelid"], axis=1, inplace=False)

    return X_train, Y_train, X_test, Y_test