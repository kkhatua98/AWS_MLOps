# See this link to understand how to write custom inference file to host Tensorflow models in SageMaker Endpoint:
# https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html#how-to-implement-the-pre-and-or-post-processing-handler-s

import os
import json
import pandas
import logging
import argparse
from io import StringIO




def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    print("Insider input_handler")
    print(data)
    print(context.request_content_type)
    if context.request_content_type == 'application/json':
        print("JSON")
        d = data.read().decode('utf-8')
        tbr = d if len(d) else ''

    if context.request_content_type == 'text/csv':
        print("CSV")
        # very simple csv handler
        tbr =  json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })
    
    print(tbr)
    print("Quiting input handler")


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    print("Inside output_handler")
    print(data.status_code)

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
    
    

# if __name__ =='__main__':
#     model_fn(model_dir = "")
