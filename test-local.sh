#!/usr/bin/env bash
set -eufx -o pipefail

./bootstrap_commons.sh

# export working directory
export ROOT_DIR=$(pwd)

# setup vault-helper aws credentials, uncomment if necessary or run in separate terminal
#   note: first we expand aliases defined in the shell ~/.bashrc so that `vault-helper` alias can be recognized
#shopt -s expand_aliases
#source ~/.bashrc

#vault-helper login -role datascience
vault-helper -c dta aws -account dataplatform-dta -rw

# refer to the correct aws token
export AWS_PROFILE=yolt-dataplatform-dta

ACCOUNT_ID=578388108212

# aws command needs the aws cli to get credentials to ECR registry
#   install following https://aws.amazon.com/cli/ ; it needs to be version >= 1.18
# note that currently we use the same docker image to every step defined below
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin "${ACCOUNT_ID}".dkr.ecr.eu-central-1.amazonaws.com

docker build -t local-tests -f test_local/Dockerfile .
