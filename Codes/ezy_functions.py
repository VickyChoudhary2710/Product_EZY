# Import libraries
import math
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import make_column_selector as selector
import warnings
label_encoder = preprocessing.LabelEncoder()




# 1. Creating Master File --
def master_data(transaction_data_path, customer_data_path, mapping_data_path, key):

    # Importing Data
    customer_data = pd.read_csv(customer_data_path)
    transaction_data = pd.read_csv(transaction_data_path)
    transaction_data["Date"] = pd.to_datetime(transaction_data["Date"])
    mapping_data = pd.read_csv(mapping_data_path)

    # Removing duplicates in customer and mapping data if any
    customer_data.drop_duplicates(subset=key, inplace=True)
    mapping_data.drop_duplicates(subset=key, inplace=True)

    # Converting joining key to str
    transaction_data[key] = transaction_data[key].astype(str)
    customer_data[key] = customer_data[key].astype(str)
    mapping_data[key] = mapping_data[key].astype(str)

    # Merging and creating master data
    # print(transaction_data.shape)
    master_data_tmp = pd.merge(transaction_data, customer_data, on=key, how="left")
    # print(master_data_tmp.shape)
    master_data = pd.merge(master_data_tmp, mapping_data, on=key, how="left")
    master_data["month"] = pd.to_datetime(master_data["Date"]).dt.month
    master_data["year"] = pd.to_datetime(master_data["Date"]).dt.year
    master_data["month"] = np.where(master_data["month"].isin([1,2,3,4,5,6,7,8,9]),'0'+master_data["month"].astype(str),master_data["month"].astype(str))
    master_data["Yearmonth"] = master_data["year"].astype(str)+ master_data["month"].astype(str)
    # print(master_data.shape)
    return master_data, customer_data


# 2. Data Cleaning --
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)


def data_manipulation(data, calc_field):

    # Identifying categorical and numerical columns
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(data)
    categorical_columns = categorical_columns_selector(data)
    # print("number of categorical columns - ", len(categorical_columns))
    # print("number of numerical columns - ", len(numerical_columns))

    # Filling missing values for categorical columns if any
    # print(data[categorical_columns].isna().sum())
    data[categorical_columns] = data[categorical_columns].fillna("miss")
    # print(data[categorical_columns].isna().sum())

    # Standardizing text fields
    for i in categorical_columns:
        data[i] = data[i].str.lower()

    # Filling missing values for numerical columns
    # print(data[numerical_columns].isna().sum())
    data[numerical_columns] = data[numerical_columns].fillna(0)
    # print(data[numerical_columns].isna().sum())
    # print(data.shape)
    return data


def outlier_removal_function(
    data,
    calc_field,
    method,
    upper_limit=None,
    lower_limit=None,
    top_perc=None,
    bottom_perc=None,
):

    if method == "IQR technique":
        # Outlier detection
        Q1 = np.percentile(data[calc_field], 25, interpolation="midpoint")
        Q3 = np.percentile(data[calc_field], 75, interpolation="midpoint")
        IQR = Q3 - Q1
        # Upper bound
        upper = Q3 + 1.5 * IQR
        # Lower bound
        lower = Q1 - 1.5 * IQR
        upper_drop = np.where(data[calc_field] >= upper)
        lower_drop = np.where(data[calc_field] <= lower)
        data.drop(upper_drop[0], inplace=True)
        data.drop(lower_drop[0], inplace=True)

    elif method == "Cutoff Values":
        if (upper_limit is None) & (lower_limit is None):
            print("Please specify cutoff values")
        elif (upper_limit is not None) & (lower_limit is None):
            upper_drop = np.where(data[calc_field] > upper_limit)
            data.drop(upper_drop[0], inplace=True)
        elif (upper_limit is None) & (lower_limit is not None):
            lower_drop = np.where(data[calc_field] < lower_limit)
            data.drop(lower_drop[0], inplace=True)
        else:
            upper_drop = np.where(data[calc_field] > upper_limit)
            lower_drop = np.where(data[calc_field] < lower_limit)
            data.drop(upper_drop[0], inplace=True)
            data.drop(lower_drop[0], inplace=True)

    elif method == "Percentile Method":
        if (top_perc is None) & (bottom_perc is None):
            print("Please percentile values")
        elif (top_perc is not None) & (bottom_perc is None):
            top_perc_value = data[calc_field].quantile(top_perc)
            upper_drop = np.where(data[calc_field] > top_perc_value)
            data.drop(upper_drop[0], inplace=True)
        elif (top_perc is None) & (bottom_perc is not None):
            bottom_perc_value = data[calc_field].quantile(bottom_perc)
            lower_drop = np.where(data[calc_field] < bottom_perc_value)
            data.drop(lower_drop[0], inplace=True)
        else:
            top_perc_value = data[calc_field].quantile(top_perc)
            bottom_perc_value = data[calc_field].quantile(bottom_perc)
            upper_drop = np.where(data[calc_field] > top_perc_value)
            lower_drop = np.where(data[calc_field] < bottom_perc_value)
            data.drop(upper_drop[0], inplace=True)
            data.drop(lower_drop[0], inplace=True)

    return data


# Data filtering based on user inputs
def data_filter_dict(data, filt_dict=None):
    if filt_dict is not None:
        for key in filt_dict:
            data = data[data[key].isin(filt_dict[key])]
    else:
        print("No filtraion criteria passed")
    return data


# Data filtering based on campaign
def data_filtration(data, campaign_start_date, campaign_end_date, pre_campaign_date):
    campaign_period_data = data[
        (data["Date"] >= campaign_start_date) & (data["Date"] < campaign_end_date)
    ]
    pre_campaign_period_data = data[data["Date"] <= pre_campaign_date]
    campaign_period_data.to_csv("campaign_period_data.csv", index=False)
    pre_campaign_period_data.to_csv("pre_campaign_period_data.csv", index=False)
    return campaign_period_data, pre_campaign_period_data

def data_grouping(data, customer_data, group_id,join_id ,metric):
    data_f = data.groupby(group_id, as_index = False)[metric].sum()
    data_f = pd.merge(data_f,customer_data, on = join_id, how = 'left')
    return data_f
    


def control_matching(method, pre_cust_df, during_cust_df, pre_trans_df, during_trans_df, flag, var_list, id_col):
    if method == "KNN":
        df = pre_cust_df.copy()
        cols = df[var_list].columns
        num_cols = list(df[var_list]._get_numeric_data().columns)
        cat_cols = list(set(cols) - set(num_cols))
        cat_cols1 = []
        for i in cat_cols:
            df[i+'_encoded'] = label_encoder.fit_transform(df[i])
            cat_cols1.append(i+'_encoded')
        var_list = num_cols
        for i in range(len(cat_cols1)):
            var_list.append(cat_cols1[i])
        
        var_list_x = []
        var_list_y = []
        for i in range(len(var_list)):
            x = var_list[i] + "_x"
            y = var_list[i] + "_y"
            var_list_x.append(x)
            var_list_y.append(y)
        n_neighbors = 1
        df_test = df[df[flag] == "test"]
        df_control = df[df[flag] == "control"]
        df_test.reset_index(drop=True, inplace=True)
        df_control.reset_index(drop=True, inplace=True)
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(pd.DataFrame(df_control[var_list]))
        distances, indices = neigh.kneighbors(pd.DataFrame(df_test[var_list]))
        indices = pd.DataFrame(indices)
        indices.rename(columns={0: "indice"}, inplace=True)
        df_match = pd.concat([df_test, indices], axis=1)
        #df_match = pd.concat([df_test, indices], axis=1)
        df_control["indice_c"] = df_control.index
        df_match2 = pd.merge(
            df_match, df_control, left_on="indice", right_on="indice_c"
        )

        
        df_c_weights = df_match2.groupby(id_col+'_y', as_index = False)[var_list_y[0]].count()
        df_c_weights.rename(columns = {id_col+'_y':id_col,var_list_y[0]: 'control_weight'  },inplace = True)
        
        pre_trans_data_test = pre_trans_df[pre_trans_df[flag] == 'test']
        pre_trans_data_control = pre_trans_df[pre_trans_df[flag] == 'control']
        pre_trans_data_control = pd.merge(pre_trans_data_control, df_c_weights, on = id_col, how = 'left')
        pre_trans_data_control.control_weight.fillna(0,inplace = True)
        pre_trans_data_control = pre_trans_data_control.loc[pre_trans_data_control.index.repeat(pre_trans_data_control.control_weight)]
        pre_trans_data_control.reset_index(inplace = True)
        
        during_trans_data_test = during_trans_df[during_trans_df[flag] == 'test']
        during_trans_data_control = during_trans_df[during_trans_df[flag] == 'control']
        during_trans_data_control = pd.merge(during_trans_data_control, df_c_weights, on = id_col, how = 'left')
        during_trans_data_control.control_weight.fillna(0,inplace = True)
        during_trans_data_control = during_trans_data_control.loc[during_trans_data_control.index.repeat(during_trans_data_control.control_weight)]
        during_trans_data_control.reset_index(inplace = True)
        
        final_data_test = pd.concat([pre_trans_data_test,during_trans_data_test],axis = 0)
        final_data_control = pd.concat([pre_trans_data_control,during_trans_data_control],axis = 0)

        
        averageby = var_list_x
        toaverage = var_list_y
        df_match3 = df_match2.groupby(averageby)[toaverage].mean().reset_index()
        c = df_match3[var_list_x].mean().reset_index(drop=True)
        t = df_match3[var_list_y].mean().reset_index(drop=True)
        s = pd.concat([t, c], axis=1)
        s.columns = ["treatment", "comparison"]
        s["variable"] = var_list
        s["difference"] = s["treatment"] - s["comparison"]
        s = s[["variable", "treatment", "comparison", "difference"]]

    
    return final_data_test, final_data_control


def plot_charts(test_data, control_data, x_axis, y_axis):
    a = test_data[x_axis]
    b = test_data[y_axis]
    fig = plt.figure(figsize = (10, 5))
    test_plot = plt.bar(a,b, color ='maroon',
        width = 0.4)
    c = control_data[x_axis]
    d = control_data[y_axis]
    fig = plt.figure(figsize = (10, 5))
    control_plot = plt.bar(c,d, color ='maroon',
        width = 0.4)

    return test_plot, control_plot

def plot_charts_avg(test_data, control_data, x_axis, y_axis):
    #using average
    test_data = test_data.groupby(x_axis,as_index = False)[y_axis].mean()
    a = test_data[x_axis]
    b = test_data[y_axis]
    fig = plt.figure(figsize = (10, 5))
    plt.plot(a, b,label = "test" )
    control_data = control_data.groupby(x_axis,as_index = False)[y_axis].mean()
    c = control_data[x_axis]
    d = control_data[y_axis]
    plt.plot(c,d, label = 'control')
    
    plt.legend()

    return plt
