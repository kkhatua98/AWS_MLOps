##

def preprocessing_function():
    import os
    import json
    import logging
    import joblib
    import traceback
    import argparse
    import numpy as np
    import pandas as pd
    import subprocess
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
    
    print(os.environ.get('SM_CHANNEL_TEST'))

    


    ###########################     Extracting the command line arguments     ########################

    parser = argparse.ArgumentParser()
    ## Adding arguments
    # Inputs
    parser.add_argument('--data_location', type=str, default="/opt/ml/processing/input/data/evaluation.csv")
    parser.add_argument('--model_location', type=str, 
                        default = "/opt/ml/processing/input/model/"
                        # default = "/opt/ml/processing/input/model/model.joblib"
                       )
    parser.add_argument('--objective_metric', type=str, default="accuracy")
    
    parser.add_argument("--best_model_location", type = str)
    parser.add_argument("--model_name", type = str)
    
    parser.add_argument("--best_training_job_container", type = str)
    parser.add_argument("--best_instance_type", type = str)

    # Outputs
    parser.add_argument('--output_location', type=str, default="/opt/ml/processing/logss")
    

    ## Parsing    
    args, _ = parser.parse_known_args()

    ###########################     Extracting the command line arguments : End     ########################
    




    ###########################     Creating the log extractor     ########################

    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f'{args.output_location}/logfile.log')
    logger.addHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    ###########################     Creating the log extractor : End     ########################
    
    
        



    try:
        current_path = os.getcwd()
        os.chdir(args.model_location)
        subprocess.run(["tar", "-xvf", args.model_location + "model.tar.gz"])
        os.chdir(current_path)
        print("Done")
        
        ## Reading datasets
        data = pd.read_csv(args.data_location)
        print("Data read.")
        model = joblib.load(args.model_location + "model.joblib")
        print("Model loaded.")
        
        X = data.drop(columns = ["Churn"])
        y = data.Churn
        
        predictions = model.predict(X)
        
        objective_metric = args.objective_metric
        if objective_metric == "accuracy":
            metric_value = accuracy_score(y, predictions)
        elif objective_metric == "precision":
            metric_value = precision_score(y, predictions)
        elif objective_metric == "recall":
            metric_value = recall_score(y, predictions)
        elif objective_metric == "f1-score":
            metric_value = f1_score(y, predictions)
        
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        conf_matrix = confusion_matrix(y, predictions)
        fpr, tpr, _ = roc_curve(y, predictions)
            
        
        report_dict = {
            "best_metric_value": metric_value, 
            "best_model": args.model_name, 
            "best_model_location":args.best_model_location, 
            "best_training_job_container":args.best_training_job_container, 
            "best_instance_type":args.best_instance_type
        }
        
        evaluation_path = "/opt/ml/processing/test/evaluation.json"
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))
        
        metrics = {
            "binary_classification_metrics": {
                "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
                "precision": {"value": precision, "standard_deviation": "NaN"},
                "recall": {"value": recall, "standard_deviation": "NaN"},
                "confusion_matrix": {
                    "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                    "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
                },
                "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
                },
            }
        }
        
        metrics_path = "/opt/ml/processing/metrics/metrics.json"
        with open(metrics_path, "w") as f:
            f.write(json.dumps(metrics))
            
        
        # joblib.dump(model, "/opt/ml/processing/model/model.joblib")  ## New line.

        logger.info("Data written to disk inside container.")


        logger.info("Preprocessing completed.")
        
        logger.removeHandler(handler)
        handler.close()
    
    except:
        var = traceback.format_exc()
        logger.error(var)
        
        logger.removeHandler(handler)
        handler.close()



if __name__ == "__main__":
    preprocessing_function()
