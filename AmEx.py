import random
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from easydict import EasyDict 

# Create a configuration dictionary
configuration = EasyDict({
    "input_dir": '/kaggle/working/',
    "seed": 40,
    "n_folds": 5,
    "target": 'target',
    "boosting_type": 'dart',
    "metric": 'binary_logloss',
    "cat_features": [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68"
    ]
})

# Set seed for reproducibility
random.seed(configuration.seed)
np.random.seed(configuration.seed)

def amex_metric(actual, predicted):
    # Create a labels array
    labels = np.column_stack((actual, predicted))
    # Sort labels based on predicted probability in descending order
    labels = labels[np.argsort(-predicted)]
    # Assign weight of 20 for negative class and 1 for positive class
    weights = np.where(labels[:, 0] == 0, 20, 1)
    # Get top 4% of the labels based on weight
    cut_off = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    # Calculate top 4% accuracy
    top_four_accuracy = np.sum(cut_off[:, 0]) / np.sum(labels[:, 0])
    # Initialize gini array
    gini = [0, 0]
    for i in [1, 0]:
        # Create labels array and sort based on predicted or actual
        labels = np.column_stack((actual, predicted))
        labels = labels[np.argsort(-predicted if i else -actual)]
        # Assign weight of 20 for negative class and 1 for positive class
        weight = np.where(labels[:, 0] == 0, 20, 1)
        # Calculate weight_random
        weight_random = np.cumsum(weight / np.sum(weight))
        # Calculate total_pos, cum_pos_found, and lorentz
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        # Calculate gini
        gini[i] = np.sum((lorentz - weight_random) * weight)
    # Return final metric
    return 0.5 * (gini[1] / gini[0] + top_four_accuracy)

def lightgbm_amex_metric(predicted, actual):
    actual = actual.get_label()
    metric_value = amex_metric(actual, predicted)
    return 'amex_metric', metric_value, True

def get_difference(data, numeric_features):
    """
    Function to calculate the difference of numeric features in a dataframe and group by 'customer_ID'
    """
    # Initialize lists to store differences and customer IDs
    differences = []
    customer_ids = []
    
    # Iterate over groups of dataframe, grouped by 'customer_ID'
    for customer_id, group in tqdm(data.groupby(["customer_ID"])):
        # Calculate the differences of numeric_features
        diff_numeric = group[numeric_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        differences.append(diff_numeric)
        customer_ids.append(customer_id)
    # Concatenate the differences and customer IDs
    differences = np.concatenate(differences, axis=0)
    # Transform to dataframe
    differences = pd.DataFrame(differences, columns=[col + "_diff1" for col in group[numeric_features].columns])
    # Add customer id
    differences["customer_ID"] = customer_ids
    return differences

def read_preprocess_data():
    """
    Function to read, preprocess, and aggregate data
    """
    # Read train data
    train = pd.read_csv("/kaggle/input/amex-default-prediction/train_data.csv")
    # Identify categorical and numerical features
    features = train.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
    categorical_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    numeric_features = [col for col in features if col not in categorical_features]
    # Aggregate numerical features by customer_ID
    train_numeric_agg = train.groupby("customer_ID")[numeric_features].agg(["mean", "std", "min", "max", "last"])
    train_numeric_agg.columns = ["_".join(x) for x in train_numeric_agg.columns]
    train_numeric_agg.reset_index(inplace=True)
    # Aggregate categorical features by customer_ID
    train_categorical_agg = train.groupby("customer_ID")[categorical_features].agg(["count", "last", "nunique"])
    train_categorical_agg.columns = ["_".join(x) for x in train_categorical_agg.columns]
    train_categorical_agg.reset_index(inplace=True)
    # Read train labels
    train_labels = pd.read_csv("/kaggle/input/amex-default-prediction/train_labels.csv")
    # Convert float64 columns to float32
    cols = list(train_numeric_agg.dtypes[train_numeric_agg.dtypes == "float64"].index)
    for col in tqdm(cols):
        train_numeric_agg[col] = train_numeric_agg[col].astype(np.float32)
    # Convert int64 columns to int32
    cols = list(train_categorical_agg.dtypes[train_categorical_agg.dtypes == "int64"].index)
    for col in tqdm(cols):
        train_categorical_agg[col] = train_categorical_agg[col].astype(np.int32)
    # Calculate differences of numeric features by customer_ID
    train_differences = get_difference(train, numeric_features)
    # Merge the aggregated features, differences, and labels
    train = train_numeric_agg.merge(
        train_categorical_agg, how="inner", on="customer_ID"
    ).merge(train_differences, how="inner", on="customer_ID").merge(
        train_labels, how="inner", on="customer_ID"
    )
    # Read test data
    test = pd.read_csv("/kaggle/input/amex-default-prediction/test_data.csv")
    # Perform the same operations on the test data
    test_numeric_agg = test.groupby("customer_ID")[numeric_features].agg(["mean", "std", "min", "max", "last"])
    test_numeric_agg.columns = ["_".join(x) for x in test_numeric_agg.columns]
    test_numeric_agg.reset_index(inplace=True)
    test_categorical_agg = test.groupby("customer_ID")[categorical_features].agg(["count", "last", "nunique"])
    test_categorical_agg.columns = ["_".join(x) for x in test_categorical_agg.columns]
    test_categorical_agg.reset_index(inplace=True)
    test_differences = get_difference(test, numeric_features)
    test = test_numeric_agg.merge(test_categorical_agg, how="inner", on="customer_ID").merge(
        test_differences, how="inner", on="customer_ID"
    )
    return train,test 


train,test = read_preprocess_data()

def train_and_evaluate(train, test):
    cat_cols = [f"{col}_last" for col in configuration.cat_features]
    float_cols = [col for col in train.select_dtypes(include=['float']).columns if 'last' in col]
    
    for col in cat_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    
    train[float_cols] = train[float_cols].round(2)
    test[float_cols] = test[float_cols].round(2)
    num_cols = [col[:-5] for col in train.columns if 'last' in col and 'round' not in col]
    for col in num_cols:
        train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
        test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
        
    float_cols = train.select_dtypes(include=['float']).columns
    train[float_cols] = train[float_cols].astype(np.float16)
    test[float_cols] = test[float_cols].astype(np.float16)
    
    features = [col for col in train.columns if col not in ['customer_ID', configuration.target]]
    
    params = {
        'objective': 'binary',
        'metric': configuration.metric,
        'boosting': configuration.boosting_type,
        'seed': configuration.seed,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
    }
    test_predictions = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))

    from sklearn.model_selection import StratifiedKFold

    import lightgbm as lightgbm
    kfold = StratifiedKFold(n_splits=configuration.n_folds, shuffle=True, random_state=configuration.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[configuration.target])):
        print(f'\nTraining fold {fold} with {len(features)} features...')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[configuration.target].iloc[trn_ind], train[configuration.target].iloc[val_ind]
        lightgbm_train = lightgbm.Dataset(x_train, y_train, categorical_feature=cat_cols)
        lightgbm_val = lightgbm.Dataset(x_val, y_val, categorical_feature=cat_cols)
        model = lightgbm.train(params, lightgbm_train, valid_sets=[lightgbm_train, lightgbm_val],
                               valid_names=['train', 'val'], num_boost_round=1000,
                               early_stopping_rounds=50, verbose_eval=50,
                               feval=lightgbm_amex_metric)
        oof_predictions[val_ind] = model.predict(x_val)
        test_predictions += model.predict(test[features]) / configuration.n_folds
        score = amex_metric(y_val,model.predict(x_val))
    score = amex_metric(train[configuration.target], oof_predictions)
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
    test_df.to_csv(f'/kaggle/working/submission.csv', index = False)

train_and_evaluate(train, test)
