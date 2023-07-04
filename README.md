# Labels Model

## Release notes 
- 21-02-2022: 
    - Added python business rules on top of the original TensorFlow model
    - UK and NL (added as additional labels) have been implemented from datascience-model-commons version 0.3.7


## Target solution architecture for model training


## Code structure

`common`: common python code used both in preprocessing and training

`config`: configuration python files: local / dta / prd

`deploy`: bash scripts for code deployment to AWS S3 and Airflow

`infrastructure`: terraform code for role and bucket creation

`orchestration`: DAGs definition and AWS operators for preprocessing and training

`preprocessing`: creating the docker image for testing locally & python scripts used in preprocessing part

`resources`: notebooks and additional useful resources used in debugging/analyzing categories model

`training`: creating the docker image for testing locally & python scripts used in training part

## Local testing

`test-local.sh`

The bash script *test-local.sh* will execute tests inside a Docker container: both for *preprocessing* and *training*. It will verify the code structure and run additional tests locally.

## Airflow local testing

Check the instruction on Confluence.

## AWS testing

`aws-dta-preprocessing.sh`

The bash script *test-dta.sh* will execute preprocessing job on AWS in DTA environment, which takes all the data needed for categories and return three parquets:
 * *preprocessed_training_data* -> parquet which we use for training
 * *preprocessed_validation_data* -> parquet which we use for validation
 * *preprocessed_test_data* -> parquet which we use to assess performance
 * *preprocessing_metadata.xml* -> preprocessing metadata

The first step in the bash script is copying the source code to s3 and then we execute it when running *ScriptProcessor* operator defined in *orchestration/python/aws_preprocess.py*. The entry point to the operator is bash script which first installs additional python requirements based on *requirements.txt* and then executes main python script.

`aws-dta-training.sh`
 
The bash script *aws-dta-training.sh* will execute training job locally using AWS data (we control this option by specifying train_instance_type). The training job takes the output of processing part and outputs a tar.gz with categories-model and training_metadata.xml inside.

In training job entry point script must be a path to a local file: train.py. We specify it in TensorFlow operator defined in orchestration/python/aws_train.py.

During training we use default tensorflow docker provided by AWS. However we also use requirements.txt based on which additional python packages are installed in the docker. It is automatically picked up in TensorFlow operator if it exists in source directory.

### Useful links for debugging AWS Sagemaker

* Check the categories-model bucket
    * Login with datascience role; replace <env> to dta or prd: `vault-helper -c <env> login --role datascience`
    * Connect to aws; replace <env> to dta or prd: `vault-helper -c <env> aws --account dataplatform-<env>`
    * Click on the link printed out in the console
    * Go to the bucket specific link; replace <env> to dta or prd: `https://s3.console.aws.amazon.com/s3/buckets/yolt-dp-<env>-datascience-categories-model/?region=eu-central-1&tab=overview`
* Only available on DTA: check the logs from Sagemaker Processing job i.e. the logs from `ScriptProcessor`: https://eu-central-1.console.aws.amazon.com/cloudwatch/home?region=eu-central-1#logStream:group=/aws/sagemaker/ProcessingJobs;streamFilter=typeLogStreamPrefix
* Only available on DTA: check the logs from Sagemaker Tensorflow job: https://eu-central-1.console.aws.amazon.com/sagemaker/home?region=eu-central-1#/jobs

## Docker used in `ScriptProcessor`

When using *ScriptProcessor* we use custom spark docker image built in separate repository for sagemaker images (https://git.yolt.io/dataplatform/sagemaker-baseimages) where we push the image to given aws environment. In order to build and push the image to AWS ECR, run the pipeline in this repository with
 * Input variable key *VAULT_ROLE*
 * Input variable value *datascience*

Then go through two CI/CD pipeline steps: *build-spark* and *push-spark-dta*

## S3 output structure
* `model_tag` - branch name from which the code is deployed
* `deploy_id` - pipeline id from which the code is deployed
* `execution_date` - date of airflow job execution, format YYYY-MM-DD-HH:MM

```
├── code
|   └─── <model_tag>
│       └─── <deploy_id>
|           └─── preprocessing
|               └─── code files
|           └─── training
|               └─── code files
├── run
│   └─── preprocessing
│       └─── <model_tag>
│           └─── <execution_date>
|               └─── output files
│   └─── training
│       └─── <model_tag>
│           └─── <execution_date>
|               └─── output files
```

Note that when training a model, we output:
- always `model.tar.gz` - model artifacts together with the metadata
- optionally `output.tar.gz` - performant model artifacts together with the metadata
which means that if the model is not performant, `output.tar.gz` will not be created.
These are two standard outputs provided by AWS: model_dir and output_data_dir (https://sagemaker.readthedocs.io/en/stable/using_tf.html#prepare-a-script-mode-training-script). We define the docker paths in the training script and then automatically AWS creates the tar.gz artifacts and store it on S3.
FIXME: It seems that there is no easy way to rename these artifacts, it would be great to have a better logic to differentiate between model vs performant model rather than having `model.tar.gz` and `output.tar.gz`

## Input/output specification
Serving output:

| COLUMN             | VALUE   | DESCRIPTION  |
| :-------------- | :------ | :----------- |
| `postprocessed_similarities`      | Array[Float]  | Array of postprocessed similarities for each category |

Serving input:

| COLUMN             | VALUE   | DESCRIPTION  |
| :-------------- | :------ | :----------- |
| `description`   | String  | Raw transaction description |
| `amount`   | Float  | Raw transaction amount |
| `transaction_type`   | String  | 'credit' or 'debit' |
| `internal_transaction`   | String  | ID for internal transactions |


Training inputs:

| TABLE | COLUMN             | VALUE   | DESCRIPTION  |
| :--- | :-------------- | :------ | :----------- |
| `users` | `id`   | String  | Unique Yolt user ID|
| `users` | `country_code`   | String  | Country |
| `test_users` | `user_id`   | String  | Unique Yolt user ID that belongs to Yolt test users|
| `account` | `id`   | String  | Account identifier; join key with `users` table |
| `account` | `deleted`   | Boolean  | Indicator whether tan account is deleted |
| `account` | `user_id`   | String  | Unique Yolt user ID|
| `transactions` | `user_id`   | String  | Unique Yolt user ID |
| `transactions` | `account_id`   | String  | Unique Yolt account ID |
| `transactions` | `transaction_id`   | String  | Unique Yolt transaction ID |
| `transactions` | `pending` | Integer  | Referring to "status", 1=REGULAR, 2=PENDING (see PENDING or REGULAR) |
| `transactions` | `date`   | String  | Date of the transaction |
| `transactions` | `description`   | String  | Transaction description |
| `transactions` | `transaction_type`   | String  | Transaction type: debit or credit |
| `transactions` | `internal_transaction`   | String  | ID for internal transactions |
| *`historical_labels_feedback` | `user_id`   | String  | Unique Yolt user ID |
| *`historical_labels_feedback` | `account_id`   | String  | Unique Yolt account ID |
| *`historical_labels_feedback` | `transaction_id`   | String  | Unique Yolt transaction ID |
| *`historical_labels_feedback` | `pending` | Integer  | Referring to "status", 1=REGULAR, 2=PENDING (see PENDING or REGULAR) |
| *`historical_labels_feedback` | `date`   | String  | Date of the transaction |
| *`historical_labels_feedback` | `feedback_time`   | String  | Date when user feedback was given |
| *`historical_labels_feedback` | `transaction_label`   | String  | Labels provided by the user |
| *`historical_categories_feedback` | `user_id`   | String  | Unique Yolt user ID |
| *`historical_categories_feedback` | `account_id`   | String  | Unique Yolt account ID |
| *`historical_categories_feedback` | `transaction_id`   | String  | Unique Yolt transaction ID |
| *`historical_categories_feedback` | `pending` | Integer  | Referring to "status", 1=REGULAR, 2=PENDING (see PENDING or REGULAR) |
| *`historical_categories_feedback` | `date`   | String  | Date of the transaction |
| *`historical_categories_feedback` | `feedback_time`   | String  | Date when user feedback was given |
| *`historical_categories_feedback` | `category`   | String  | Category provided by the user |
| `user_single_feedback_created` | `id.userId`   | String  | Unique Yolt user ID |
| `user_single_feedback_created` | `id.accountId`   | String  | Unique Yolt account ID |
| `user_single_feedback_created` | `id.transactionId`   | String  | Unique Yolt transaction ID |
| `user_single_feedback_created` | `id.pendingType` | Integer  | Referring to "status", 1=REGULAR, 2=PENDING (see PENDING or REGULAR) |
| `user_single_feedback_created` | `id.localDate`   | String  | Date of the transaction |
| `user_single_feedback_created` | `time`   | String  | Date when user feedback was given |
| `user_single_feedback_created` | `fact.category`   | String  | Category provided by the user |
| `user_multiple_feedback_created` | `id.userId`   | String  | Unique Yolt user ID |
| `user_multiple_feedback_created` | `id.accountId`   | String  | Unique Yolt account ID |
| `user_multiple_feedback_created` | `id.transactionId`   | String  | Unique Yolt transaction ID |
| `user_multiple_feedback_created` | `id.pendingType` | Integer  | Referring to "status", 1=REGULAR, 2=PENDING (see PENDING or REGULAR) |
| `user_multiple_feedback_created` | `id.localDate`   | String  | Date of the transaction |
| `user_multiple_feedback_created` | `time`   | String  | Date when user feedback was given |
| `user_multiple_feedback_created` | `fact.category`   | String  | Category provided by the user |
| `user_multiple_feedback_applied` | `id.userId`   | String  | Unique Yolt user ID |
| `user_multiple_feedback_applied` | `id.accountId`   | String  | Unique Yolt account ID |
| `user_multiple_feedback_applied` | `id.transactionId`   | String  | Unique Yolt transaction ID |
| `user_multiple_feedback_applied` | `id.pendingType` | Integer  | Referring to "status", 1=REGULAR, 2=PENDING (see PENDING or REGULAR) |
| `user_multiple_feedback_applied` | `id.localDate`   | String  | Date of the transaction |
| `user_multiple_feedback_applied` | `time`   | String  | Date when user feedback was given |
| `user_multiple_feedback_applied` | `fact.category`   | String  | Category provided by the user |

### Notes on *`historical_categories_feedback`

`historical_categories_feedback` is a static table which contains:

* `external_provider_categories`: a set of transactions with a category provided by external providers in order to
have a good starting point to train a model before we collected enough user feedback

* `reconstructed_user_feedback_before_20181107`: reconstructed user feedback before we started collecting data science
events

* `synthetic_feedback_new_categories`: a set of transactions with a new category assigned based on a set of business
rules; the table was created when Yolt decided to release this set of new categories to the users: "Gifts", "Kids",
"Takeaway", "Petrol", "Rent", "Mortgage", "Utilities", "Vehicle", "Pets", "Savings"

In order to check the details how each table was generated, check out the notebooks in `resources/notebooks/historical_feedback`.

Labels are created in same manner

