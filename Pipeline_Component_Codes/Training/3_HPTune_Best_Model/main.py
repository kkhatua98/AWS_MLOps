# Follow this link to write the logis on how to get details of a hyperparameter tuning job.
# https://sagemaker-examples.readthedocs.io/en/latest/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.html

def main(event, context):
    import boto3
    import os
    region = boto3.Session().region_name
    sage_client = boto3.Session().client("sagemaker")

    best_model_location = ''
    best_metric_value = 0
    best_model = ''

    try:
        ## You must have already run a hyperparameter tuning job to analyze it here.
        ## The Hyperparameter tuning jobs you have run are listed in the Training section on your SageMaker dashboard.
        ## Copy the name of a completed job you want to analyze from that list.
        ## For example: tuning_job_name = 'mxnet-training-201007-0054'.
        tuning_job_name = event["tuning_job_name"]


        # Get the results from hyper parameter tuning job
        tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
            )

        # Get the best training job's metric value
        best_metric_value = tuning_job_result["BestTrainingJob"]["FinalHyperParameterTuningJobObjectiveMetric"]["Value"]


        # Getting best training job's model location
        best_training_job_name = tuning_job_result["BestTrainingJob"]["TrainingJobName"]
        best_training_job_result = sage_client.describe_training_job(TrainingJobName=best_training_job_name)
        best_model_location = best_training_job_result['ModelArtifacts']["S3ModelArtifacts"]



        # The following few lines will be necessary if we are trying to keep show all the results of the hyper parameter tuning job.
        # objective = tuning_job_result["HyperParameterTuningJobConfig"]["HyperParameterTuningJobObjective"]
        # is_minimize = objective["Type"] != "Maximize"

        # tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
        # full_df = tuner.dataframe()

        # df = full_df[full_df["FinalObjectiveValue"] > -float("inf")]
        # if len(df) > 0:
        #     df = df.sort_values("FinalObjectiveValue", ascending=is_minimize)
        # df.to_csv()
    
    except:
#         model_and_metrices = event["model_and_metrices"]
#         models = [current["model"] for current in model_and_metrices]
#         metrices = [current["metrics"] for current in model_and_metrices]
#         model_locations = [current["best_model_location"] for current in model_and_metrices]
        
#         best_metric_value = max(metrices)

#         best_model = models[metrices.index(best_metric_value)]
#         best_model_location = model_locations[metrices.index(best_metric_value)]
        
        n = event['n']
        metrices = [event[f"metrics{i}"] for i in range(n)]
        
        best_metric_value = max(metrices)
        max_index = metrices.index(best_metric_value)
        best_model = event[f"model{max_index}"]
        best_model_location = event[f"best_model_location{max_index}"]
        


    
    
    return {"best_model_location":best_model_location, "best_metric_value":best_metric_value, "best_model":best_model}