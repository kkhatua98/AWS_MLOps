def lambda_handler(event, context = None):
    import boto3
    import sagemaker
    import json
    
    with open("config.json") as file:
        build_parameters = json.load(file)
    
    session = sagemaker.Session(default_bucket = "demo-output-bucket")
    role = sagemaker.get_execution_role()
    
    # Create a low-level SageMaker service client.
    client = boto3.client('sagemaker', region_name=region)
    
    # Obtain inputs from event
    
    model_package_group_name = build_parameters["model_package_group_name"]
    
    
    # Get the latest approved model
    model_packages = client.list_model_packages(ModelPackageGroupName = model_package_group_name)["ModelPackageSummaryList"]
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
    endpoint_name = build_parameters["endpoint_name"]
    try:
        response = client.describe_endpoint_config(
            EndpointConfigName=endpoint_name
        )
        model_name = response["ProductionVariants"][0]["ModelName"]
        
        # Deleting endpoint
        response = client.delete_endpoint(
            EndpointName=endpoint_name
        )
        
        # Deleting endpoint config
        response = client.delete_endpoint_config(
            EndpointConfigName=endpoint_name
        )
        
        # Deleting model
        response = client.delete_model(
            ModelName=model_name
        )
    except:
        pass
    
    
    

    
    from sagemaker.model import Model
    model = Model(
        entry_point = "Evaluation.py",
        source_dir = "code", 
        role = role,
        image_uri = image_url,
        model_data = model_data_url
    )
    
    predictor = model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name = endpoint_name)