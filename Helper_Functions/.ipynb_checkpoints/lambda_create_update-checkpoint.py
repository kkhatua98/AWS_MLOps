import json
import boto3
import subprocess
from sagemaker import get_execution_role


role = get_execution_role()
print(role)

## Loading the configurations from config.json file.
with open("config.json") as file:
    build_parameters = json.load(file)


# Listing Lambda functions
client = boto3.client('lambda')
response = client.list_functions()
functions = [func["FunctionName"] for func in response["Functions"]]


## Creating/Updating endpoint updater lambda function
# Step 1: Creating a zip file with the code and uploading it to S3.
subprocess.run(["cp", "Lambda_Functions/Update_Endpoint/update_endpoint.py", "."])
subprocess.run(["zip", '-r', "endpoint_updater_lambda.zip", "update_endpoint.py", "config.json"])
subprocess.run(["aws", "s3", "cp", "endpoint_updater_lambda.zip", f"s3://{build_parameters['input_bucket']}/codes/Lambda/"])

# Step 2: Creating/Updating Lambda function with the zip file uploaded to S3.
lambda_function_name = build_parameters["endpoint_updater_lambda_function_name"]
if lambda_function_name not in functions:
    response = client.create_function(
        Code={
            'S3Bucket':build_parameters["input_bucket"],
            'S3Key':'codes/Lambda/endpoint_updater_lambda.zip',
        },
        Description='Update churn scoring endpoint',
        FunctionName=lambda_function_name,
        Handler='update_endpoint.updater',
        Publish=True,
        Role = build_parameters["role_given_to_lambda"]
        Runtime='python3.7'
    )
    print(response)
else:
    response = client.update_function_code(
        FunctionName=lambda_function_name,
        S3Bucket=build_parameters["input_bucket"],
        S3Key='codes/endpoint_updater_lambda.zip'
    )
    print(response)


    
## Creating/Updating monitoring output notifier lambda function
# Step 1: Creating a zip file with the code and uploading it to S3.
subprocess.run(["cp", "Lambda_Functions/Monitoring_Result_Notifier/output_notifier.py", "."])
subprocess.run(["zip", '-r', "monitoring_lambda_codes.zip", "output_notifier.py"])
subprocess.run(["aws", "s3", "cp", "monitoring_lambda_codes.zip", f"s3://{build_parameters['input_bucket']}/codes/Lambda/"])

# Step 2: Creating/Updating Lambda function with the zip file uploaded to S3.
output_notifier_lambda_function_name = "model_performance_notification"
if monitoring_lambda_function_name not in functions:
    response = client.create_function(
        Code={
            'S3Bucket':build_parameters["input_bucket"],
            'S3Key':'codes/Lambda/monitoring_lambda_codes.zip',
        },
        Description='Notify monitoring pipeline output through mail',
        FunctionName="model_performance_notification",
        Handler="output_notifier.lambda_handler",
        Publish=True,
        Role=build_parameters["role_given_to_lambda"],
        Runtime='python3.7'
    )
    print(response)
else:
    response = client.update_function_code(
        FunctionName="model_performance_notification",
        S3Bucket=build_parameters["input_bucket"],
        S3Key='codes/Lambda/monitoring_lambda_codes.zip'
    )
    print(response)


