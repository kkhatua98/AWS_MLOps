
def lambda_handler(event, context):
    import boto3
    import subprocess
    subprocess.run(["pip", "install", "sagemaker"])
    
    # Obtain inputs from event
    region=event["region"]
    model_name = event["model_name"]
    role = event["role"]
    
    
    # Create a low-level SageMaker service client.
    client = boto3.client('sagemaker', region_name=region)
    
    
    # Get the latest approved model
    model_packages = client.list_model_packages(ModelPackageGroupName = event["model_package_group_name"])["ModelPackageSummaryList"]
    for model_package in model_packages:
        if model_package["ModelApprovalStatus"] == "Approved":
            latest_package_arn = model_package["ModelPackageArn"]
            break
    
    
    # Get details of the latest approved model
    latest_package_details = client.describe_model_package(ModelPackageName=latest_package_arn)
    image_url = latest_package_details['InferenceSpecification']['Containers'][0]['Image']
    model_data_url = latest_package_details['InferenceSpecification']['Containers'][0]["ModelDataUrl"]
    instance_type = latest_package_details["InferenceSpecification"]["SupportedTransformInstanceTypes"][0]
    
    
    # Deleting model, endpoint config and endpoint if already present.
    try:
        # Deleting model
        response = client.delete_model(
            ModelName=model_name
        )
        
        # Deleting endpoint config
        response = client.delete_endpoint_config(
            EndpointConfigName=f"{model_name}-config"
        )
        
        # Deleting endpoint
        response = client.delete_endpoint(
            EndpointName=f"{model_name}-endpoint"
        )
    except:
        pass
    
    from sagemaker.model import Model
    inference_model = Model(image_uri = image_url, 
                                    
                                    ## -------- ##
                                    source_dir = f"s3://{build_parameters['input_bucket']}/codes/endpoint_scoring.tar.gz",
                                    entry_point = f"{build_parameters['endpoint_scoring_code_location'].split('/')[-1]}",
                                    # entry_point="../" + build_parameters["scoring_code_loaction"], 
                                    ## -------- ##
                                    
                                    model_data = latest_package_details['InferenceSpecification']['Containers'][0]["ModelDataUrl"], 
                                    role = role,
                                    sagemaker_session = sagemaker_session,
                                    
                                    env = {"target_column":build_parameters["target_column"],
                                           "feature_selection_file_location":f"s3://{build_parameters['input_bucket']}/Feature_Selection.csv",
                                           "log_location":"/opt/ml/processing/logss"
                                          }
                                   )
    
    # Creating model
    create_model_response = client.create_model(
        ModelName = model_name,
        ExecutionRoleArn = role,
        PrimaryContainer = {
            'Image': image_url,
            # 'ModelDataUrl': "s3://demo-output-bucket/Training_Pipeline_Output/2023-08-02T06:55:59.009Z/HPTuningOutputs/Decision_Tree/bqbpzw5k3d15-hptuning-U0H0Wl6oKU-001-c42f316a/output/model.tar.gz",
            "ModelDataUrl":model_data_url
        }
    )
    
    # Creating endpoint config
    endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName=f"{model_name}-config", # You will specify this name in a CreateEndpoint request.
        # List of ProductionVariant objects, one for each model that you want to host at this endpoint.
        ProductionVariants=[
            {
                "VariantName": "variant1", # The name of the production variant.
                "ModelName": model_name, 
                "InstanceType": instance_type, # Specify the compute instance type.
                "InitialInstanceCount": 1 # Number of instances to launch initially.
            }
        ]
    )
    
    # Creating endpoint
    response = client.create_endpoint(
        EndpointName=f"{model_name}-endpoint",
        EndpointConfigName=f"{model_name}-config"
    )
    
            