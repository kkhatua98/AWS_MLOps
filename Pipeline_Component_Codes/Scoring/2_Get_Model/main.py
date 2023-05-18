def main(event, context):
    import boto3
    region = boto3.Session().region_name
    client = boto3.client('sagemaker')
    
    model_packages = client.list_model_packages(ModelPackageGroupName = event["package_group"])