from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import mlflow

def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path) 
    return data

def get_classes(target_data: pd.Series) -> List[str]:
    return list(target_data.unique())

def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}

#seperating categorical data
def seperating_categorical(data: pd.DataFrame) -> pd.DataFrame:
    sep_categorical = data.select_dtypes(include=['object'])
    return sep_categorical

def encoding(data:pd.DataFrame) -> pd.DataFrame:
    cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}
    data['cut'] = data['cut'].apply(lambda x : cut_encoder[x])
    
    color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}
    data['color'] = data['color'].apply(lambda x : color_encoder[x])
    
    clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}
    data['clarity'] = data['clarity'].apply(lambda x : clarity_encoder[x])

    return data

#seperating numerical data
def seperating_numerical(data:pd.DataFrame) -> pd.DataFrame:
    sep_numerical = data.select_dtypes(include=['int64', 'float64'])
    return sep_numerical

def get_scaler(data: pd.DataFrame) -> Any:
    # scaling the numerical features
    scaler = StandardScaler()
    scaler.fit(data)
    
    return scaler

def rescale_num_data(data: pd.DataFrame, scaler: Any) -> pd.DataFrame:    
    # scaling the numerical features
    # column names are (annoyingly) lost after Scaling
    # (i.e. the dataframe is converted to a numpy ndarray)
    num_data_rescaled = pd.DataFrame(scaler.transform(data), 
                                columns = data.columns, 
                                index = data.index)
    return num_data_rescaled

def concat_df(data:pd.DataFrame,data1:pd.DataFrame) -> pd.DataFrame:
    concated_df= pd.concat([data,data1], axis=1)
    return  concated_df

def find_best_model(X_train:pd.DataFrame,y_train:pd.Series,estimator:Any,parameters:List)->Any:
    #Enabling automatic MLFLOW logging for Scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)
    
    with mlflow.start_run():
        clf=GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring='neg_mean_absolute_error',
            cv=5,
            return_train_score=True,
            verbose=1
        )
        clf.fit(X_train,y_train)
        
        #Disabling autologging
        mlflow.sklearn.autolog(disable=True)
        
        return clf

# Workflow
def main(path: str):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diamond Price Prediction")

    # Define Parameters
    TARGET_COL = 'price'
    TEST_DATA_RATIO = 0.2
    DATA_PATH = path

    dataframe=load_data(path=DATA_PATH)

    # Identify Target Variable
    target_data = dataframe[TARGET_COL]
    input_data = dataframe.drop([TARGET_COL], axis=1)

    # Split the Data into Train and Test
    train_test_dict = split_data(input_=input_data, output_=target_data, test_data_ratio=TEST_DATA_RATIO)

    #Preprocessing X_train
    Numerical_train_df=seperating_numerical(train_test_dict['X_TRAIN'])
    Categorical_train_df=seperating_categorical(train_test_dict['X_TRAIN'])
    X_train_cat_le=encoding(Categorical_train_df)
    concated_df1=concat_df(Numerical_train_df,X_train_cat_le)
    scaler = get_scaler(concated_df1)
    X_train_transformed = rescale_num_data(data=concated_df1, scaler=scaler)


    #Preprocessing X_test
    Numerical_test_df=seperating_numerical(train_test_dict['X_TEST'])
    Categorical_test_df=seperating_categorical(train_test_dict['X_TEST'])
    X_test_cat_le=encoding(Categorical_test_df)
    concated_df2=concat_df(Numerical_test_df,X_test_cat_le)
    scaler = get_scaler(concated_df2)
    X_test_transformed = rescale_num_data(data=concated_df2, scaler=scaler)

    # Model Training
    ESTIMATOR=KNeighborsRegressor()
    HYPERPARAMETERS = [{'n_neighbors':[i for i in range(1, 31)], 'p':[1, 2]}]
    regressor=find_best_model(X_train_transformed,train_test_dict['Y_TRAIN'],ESTIMATOR,HYPERPARAMETERS)
    print(regressor.best_params_)
    print(regressor.score(X_test_transformed,train_test_dict['Y_TEST']))
    
# Run the main function
main(path='./data/diamonds.csv')