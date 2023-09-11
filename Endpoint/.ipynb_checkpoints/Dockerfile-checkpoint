# https://stackoverflow.com/questions/73607950/docker-file-aws-lambda-local-class-import-error
# https://stackoverflow.com/questions/51409209/dockerfile-copy-files-from-amazon-s3-or-another-source-that-needs-credentials

# How to keep subdirectory structure inside Docker image: https://stackoverflow.com/questions/30215830/dockerfile-copy-keep-subdirectory-structure
# How to use custom Docker Images with Lambda Function: https://docs.aws.amazon.com/lambda/latest/dg/python-image.html

FROM public.ecr.aws/lambda/python:3.9

# Copy function code
COPY Function_Contents/ ${LAMBDA_TASK_ROOT}/

# install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
RUN pip3 install --upgrade "protobuf<=3.20.1" --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]
