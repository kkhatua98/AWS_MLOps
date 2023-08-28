

def model_fn(model_dir):
    """
    Function to load model.
    """
    import os
    import pandas
    import logging
    import argparse
    import joblib


    ## Creating a logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info("Inference started.")
    
    
    
    print("Files are")
    dir_list = os.listdir("/opt/ml/model")
    print(dir_list)
    
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor



def input_fn(request_body, request_content_type):
    """
    An input_fn that loads a pickled numpy array
    """
    import pandas
    from io import StringIO
    
    print("Request body")
    print(request_body)
    print(type(request_body))
    print("String IO request body")
    # print(StringIO(request_body.decode("utf-8"))) # Batch
    print(StringIO(request_body)) # Endpoint
    
    # df = pandas.read_csv(StringIO(request_body.decode("utf-8"))) # Batch
    df = pandas.read_csv(StringIO(request_body), header = None) # Endpoint
    df.columns = ["Account length", "Number vmail messages", "Total day minutes", "Total day calls", "Total eve minutes", "Total eve calls", "Total night minutes", 
                  "Total night calls", "Total intl minutes", "Total intl calls", "Customer service calls", "Total_minutes", "Total_calls", "Minutes_per_call_overall", 
                  "Minutes*call_overall", "Minutes_per_call_int", "Minutes*call_int", "Minutes_per_call_day", "Minutes*call_day", "Minutes_per_call_eve", "Minutes*call_eve",
                  "Minutes_per_call_night", "Minutes*call_night", "Total_charge", "Day_minutes_per_customer_service_calls", "Day_minutes*customer_service_calls", 
                  "Total_day_minutes_wholenum", "Total_day_minutes_decimalnum", "Total_minutes_wholenum", "Total_minutes_decimalnum", "Voice_and_Int_plan", "Only_Int_plan", 
                  "Only_vmail_plan", "No_plans", "State_AL", "State_AR", "State_AZ", "State_CA", "State_CO", "State_CT", "State_DC", "State_DE", "State_FL", "State_GA", "State_HI", 
                  "State_IA", "State_ID", "State_IL", "State_IN", "State_KS", "State_KY", "State_LA", "State_MA", "State_MD", "State_ME", "State_MI", "State_MN", "State_MO", "State_MS", 
                  "State_MT", "State_NC", "State_ND", "State_NE", "State_NH", "State_NJ", "State_NM", "State_NV", "State_NY", "State_OH", "State_OK", "State_OR", "State_PA", "State_RI",
                  "State_SC", "State_SD", "State_TN", "State_TX", "State_UT", "State_VA", "State_VT", "State_WA", "State_WI", "State_WV", "State_WY", "Area code_415", "Area code_510", 
                  "International plan_Yes", "Voice mail plan_Yes", "Account_length_bins_q2", "Account_length_bins_q3", "Account_length_bins_q4", "zero_vmails_Yes", 
                  "Customer_service_calls_bins_q2", "Customer_service_calls_bins_q3", "Customer_service_calls_bins_q4"]
    print(df.head())
    return df



def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    # import xgboost as xgb
    from sklearn.metrics import accuracy_score
    import pathlib
    import json
    import os
    import pandas
    import numpy
    
    
    prediction_probabilities = model.predict_proba(input_data)
    prediction_array = model.predict(input_data)
    # prediction_dataframe = pandas.DataFrame(numpy.vstack((prediction_array, prediction_probabilities)).T)
    constant_array = numpy.full(prediction_array.shape, 1)
    prediction_dataframe = pandas.DataFrame(numpy.vstack((prediction_array, constant_array)).T)
    
    
    return prediction_dataframe