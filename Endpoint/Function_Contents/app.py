
def lambda_handler(event, context = None):
    import boto3
    import sagemaker
    import json
    print("Imported libraries.")
    with open("config.json") as file:
        build_parameters = json.load(file)
    
    session = sagemaker.Session(default_bucket = build_parameters["otuput_bucket"])
    region = boto3.Session().region_name
    role = sagemaker.get_execution_role()
    
    # Create a low-level SageMaker service client.
    client = boto3.client('sagemaker', region_name=region)
    
    # Obtain inputs from event
    model_package_group_name = build_parameters["inputs"]["model_package_group_name"]
    
    
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
    endpoint_name = build_parameters["inputs"]["endpoint_name"]
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
    from sagemaker.tensorflow import TensorFlowModel
    
    sklearn_models = ["Decision_Tree", "Logistic_Regression"]
    if any([model in model_data_url for model in sklearn_models]):
        model = Model(
            entry_point = "Evaluation.py",
            source_dir = "code/Sklearn_Endpoint", 
            role = role,
            image_uri = image_url,
            model_data = model_data_url
        )
        
        
        
    elif "Tensorflow" in model_data_url:
        # See the following links to deploy a Tensorflow model:
        # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html#deploying-directly-from-model-artifacts
        # https://github.com/mobassir94/Deploy-trained-TensorFlow-2.x-models-using-Amazon-SageMaker/blob/main/How_to_deploy_custom_tf_2_x_model_using_aws_sagemaker.ipynb
        model = TensorFlowModel(
            model_data=model_data_url, 
            role = role, 
            framework_version = "1.15.2", 
            entry_point = "inference.py", 
            source_dir = "code/Tensorflow_Endpoint"
        )
        
        
    
    predictor = model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name = endpoint_name)
    
    
    # In the following way we have to get predictions from the Sklearn endpoint.
    # See this link to get the idea on how to get predictions from a SageMaker Scikit-Learn Endpoint: 
    # https://stackoverflow.com/questions/63568274/how-to-use-serializer-and-deserializer-in-sagemaker-2
    # from sagemaker.deserializers import JSONDeserializer
    # from sagemaker.serializers import CSVSerializer
    # predictor = sagemaker.predictor.Predictor(endpoint_name = "churn-endpoint-12345",
    #                                           serializer = CSVSerializer(),
    #                                           # content_type  = "text/csv"
    #                                          )
    # import pandas as pd
    # s3 = pd.read_csv("s3://demo-output-bucket/Training_Pipeline_Output/2023-08-02T06:55:59.009Z/TrainProcessingOutput/test.csv")
    # s3 = s3.drop(columns = ["Churn"])
    # result = predictor.predict(s3)
    # print(result)
    
    
    
    # In the following way we have to get predictions from the Tensorflow endpoint.
    # See the following link to understand how to get predictions from the endpoint:
    # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html#making-predictions-against-a-sagemaker-endpoint
    # from sagemaker.serializers import JSONSerializer
    # predictor = sagemaker.predictor.Predictor(endpoint_name = "tensorflow-inference-fashion-mnist-111",
    #                                           serializer = JSONSerializer(),
    #                                           # content_type  = "text/csv"
    #                                          )
    # import numpy as np
    # random_array = np.random.randn(28, 28)
    # random_list = random_array.tolist()
    
    # inputs= {'instances': random_list}      # This formation is important.
    # result = predictor.predict(inputs)
    # print(result)
    

if __name__ == "__main__":
    lambda_handler({})