#!/bin/sh

aws s3api get-object --bucket demo-inputs-bucket --key Lambda_Code.zip Lambda_Code.zip
unzip Lambda_Code.zip -d "${LAMBDA_TASK_ROOT}"

exec "$@"