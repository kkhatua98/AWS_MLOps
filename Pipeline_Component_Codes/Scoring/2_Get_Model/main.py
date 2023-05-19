def main(event, context):
    import boto3
    region = boto3.Session().region_name
    client = boto3.client('sagemaker')
    
    model_packages = client.list_model_packages(ModelPackageGroupName = event["model_package_group_name"])["ModelPackageSummaryList"]
    for model_package in model_packages:
        if model_package["ModelApprovalStatus"] == "Approved":
            latest_package_arn = model_package["ModelPackageArn"]
            break
    
    latest_package_details = client.describe_model_package(ModelPackageName=latest_package_arn)
    image_url = latest_package_details['InferenceSpecification']['Containers'][0]['Image']
    model_data_url = latest_package_details['InferenceSpecification']['Containers'][0]["ModelDataUrl"]
    instance_type = latest_package_details["InferenceSpecification"]["SupportedTransformInstanceTypes"][0]
    
    return {"image_uri":image_url, "model_data_uri":model_data_url, "instance_type":instance_type}