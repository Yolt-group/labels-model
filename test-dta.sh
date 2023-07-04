#!/usr/bin/env bash
set -eufx -o pipefail

./bootstrap_commons.sh

#vault-helper login --role datascience
vault-helper -c dta aws --account dataplatform-dta -rw

# refer to the correct aws token
export AWS_PROFILE=yolt-dataplatform-dta

aws s3 sync test/resources/data/ s3://yolt-dp-dta-datascience-labels-model/input/

./datascience-model-commons/scripts/local_aws_dta_job_spark.sh labels preprocessing
./datascience-model-commons/scripts/local_aws_dta_job_tensorflow.sh labels training
