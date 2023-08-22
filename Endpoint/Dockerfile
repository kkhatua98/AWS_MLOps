FROM public.ecr.aws/lambda/python:3.9

# Copy function code
COPY Function_Contents/ ${LAMBDA_TASK_ROOT} 
COPY requirements.txt  ${LAMBDA_TASK_ROOT} 

# install dependencies
RUN pip3 install --user -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lamdba_handler.lambda_handler" ]