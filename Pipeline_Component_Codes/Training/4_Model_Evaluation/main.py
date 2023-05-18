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
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    
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
            
        
        report_dict = {"best_metric_value": metric_value, "best_model": args.model_name, "best_model_location":args.best_model_location}
        
        evaluation_path = "/opt/ml/processing/test/evaluation.json"
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))
            
        
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
