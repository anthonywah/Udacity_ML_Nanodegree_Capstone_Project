import sagemaker
from sagemaker.session import s3_input
import numpy as np
import pandas as pd
import sys
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch import PyTorchModel
from data_processing import *
import boto3


class SagemakerEstimatorHelper:
    def __init__(self,
                 target_algorithm, 
                 output_path,
                 **kwargs):
        """ Helper class containing 3 algorithms for convenience
        
        @param target_algorithm: Anyone from 'neural_network', 'xgboost', 'linear_learner'
        @param output_path: Path for estimators to output their results
        @param **kwargs: All keyword arguments will be passed to the specified target_algorithm
        @return: None
        """
        assert target_algorithm in ['neural_network', 'xgboost', 'linear_learner']
        print(f'Start constructing Sagemeker Estimator <{target_algorithm}> Instance')
        self.target_algorithm = target_algorithm
        self.output_path = output_path
        self.sagemaker_session = sagemaker.Session()
        self.sagemaker_role = sagemaker.get_execution_role()
        self.sagemaker_bucket = self.sagemaker_session.default_bucket()
        self.data_path_dict = {'s3': {}, 'local': {}}
        self.predictor = None
        for k, v in kwargs.items():
            setattr(self, k, v)
        if target_algorithm == 'neural_network':
            self.estimator = sagemaker.pytorch.PyTorch(
                entry_point=self.train_entry_point, 
                source_dir=self.source_dir, 
                role=self.sagemaker_role, 
                framework_version='1.0', 
                py_version='py3', 
                train_instance_count=1,  
                train_instance_type='ml.c4.xlarge', 
                output_path=output_path, 
                sagemaker_session=self.sagemaker_session, 
                hyperparameters=self.hyperparameters
            )
        elif target_algorithm == 'xgboost':
            self.container = sagemaker.amazon.amazon_estimator.get_image_uri(self.sagemaker_session.boto_region_name, 'xgboost', '1.0-1')
            self.estimator = sagemaker.estimator.Estimator(
                image_uri=self.container, 
                role=self.sagemaker_role, 
                train_instance_count=1, 
                train_instance_type='ml.m4.xlarge', 
                output_path=output_path, 
            )
        elif target_algorithm == 'linear_learner':
            self.estimator = sagemaker.LinearLearner(
                role=self.sagemaker_role, 
                instance_count=1, 
                instance_type='ml.c4.xlarge', 
                predictor_type='regressor', 
                output_path=output_path, 
                sagemaker_session=self.sagemaker_session, 
                hyperparameters=self.hyperparameters
            )
        else:
            raise Exception("UnexpectedAlgorithm")
        print('{} instance constructed'.format(target_algorithm.replace("_", " ").title()))
        return
    
    def upload_data(self, 
                    target_df, 
                    target_df_type,  # 'train' or 'test'
                    data_dir='processed_data', 
                    prefix='sagemaker/capstone_capstone', 
                    force_update=True):
        print(f'Start uploading data to s3')
        self.data_dir = data_dir
        self.prefix = prefix
        
        # Save csv to be uploaded
        data_path = os.path.join(data_dir, f'{target_df_type}.csv')
        if force_update or not os.path.exists(data_path):
            save_df(input_df=target_df, save_path=data_path)
        else:
            print(data_path, 'exists already')

        # Upload to S3, if object doesn't exist
        if force_update or os.path.join(prefix, f'{target_df_type}.csv') not in [obj.key for obj in boto3.resource('s3').Bucket(bucket).objects.all()]:
            s3_input_data_path = self.sagemaker_session.upload_data(path=data_dir, bucket=self.sagemaker_bucket, key_prefix=prefix)
            print('Uploaded to', s3_input_data_path)
        else:
            s3_input_data_path = os.path.join(f's3://{self.sagemaker_bucket}', prefix, f'{target_df_type}.csv')
            print(s3_input_data_path, 'exists already')
        self.data_path_dict['s3'][target_df_type] = s3_input_data_path
        self.data_path_dict['local'][target_df_type] = data_path
        return
    
    def set_hyperparameters(self, input_hyperparameters):
        assert self.target_algorithm == 'xgboost'
        self.estimator.set_hyperparameters(**input_hyperparameters)
        return
    
    def est_fit(self):
        print(f'Start training Estimator {self.target_algorithm}')
        if self.target_algorithm == 'neural_network':
            self.estimator.fit({'train': self.data_path_dict['s3']['train']})
        elif self.target_algorithm == 'xgboost':
            s3_input_train = s3_input(s3_data=self.data_path_dict['s3']['train'], content_type='text/csv')
            s3_input_validation = s3_input(s3_data=self.data_path_dict['s3']['test'], content_type='text/csv')
            self.estimator.fit({'train': s3_input_train, 'validation': s3_input_validation})
        elif self.target_algorithm == 'linear_learner':
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            train_features = train_df.loc[:, train_df.columns[1:]].to_numpy()
            train_labels = train_df.loc[:, train_df.columns[0]].to_numpy()
            train_x_np = train_features.astype('float32')
            train_y_np = train_labels.astype('float32')
            formatted_train_data = self.estimator.record_set(train_x_np, labels=train_y_np)
            self.estimator.fit(formatted_train_data)
        else:
            raise Exception("UnexpectedAlgorithm")
        return
    
    def deploy(self):
        print(f'Start deploying {self.target_algorithm} endpoint')
        if self.target_algorithm == 'neural_network':
            model = PyTorchModel(model_data=self.estimator.model_data,
                                 role=self.sagemaker_role,
                                 framework_version='1.0',
                                 py_version='py3',
                                 entry_point=self.predict_entry_point,
                                 source_dir=self.source_dir)
            self.predictor = model.deploy(initial_instance_count=1, 
                                          instance_type='ml.t2.medium')
        elif self.target_algorithm == 'xgboost':
            self.predictor = self.estimator.deploy(initial_instance_count=1, 
                                                   instance_type='ml.m4.xlarge')
            self.predictor.content_type = 'text/csv'
            self.predictor.serializer = sagemaker.predictor.csv_serializer
        elif self.target_algorithm == 'linear_learner':
            self.predictor = self.estimator.deploy(initial_instance_count=1, 
                                                   instance_type='ml.t2.medium')
        else:
            raise Exception("UnexpectedAlgorithm")
        return
    
    def predict(self, input_x_df):  # input_x_df should contain no label column
        assert self.predictor is not None
        if self.target_algorithm == 'neural_network':
            predictions = np.squeeze(self.predictor.predict(input_x_df))
        elif self.target_algorithm == 'xgboost':
            predictions = self.predictor.predict(input_x_df.values).decode('utf-8')
            predictions = np.fromstring(predictions, sep=',')
        elif self.target_algorithm == 'linear_learner':
            predictions = self.predictor.predict(input_x_df.to_numpy().astype('float32'))
            predictions = np.array([x.label['score'].float32_tensor.values[0] for x in predictions])
        else:
            raise Exception("UnexpectedAlgorithm")
        return predictions
    
    def predict_in_chunks(self, input_x_df, chunk_size=2000):  # input_x_df should contain no label column
        prev_i = None
        input_x_df = input_x_df.reset_index(drop=True)
        res_df = pd.DataFrame()
        data_len = input_x_df.shape[0]
        for i in range(0, data_len, chunk_size):
            if prev_i is None:
                prev_i = 0
                continue
            preds = self.predict(input_x_df.loc[range(prev_i, i), :])
            res_df = res_df.append(pd.DataFrame({'prediction': preds})).reset_index(drop=True)
            pct_ind = int((1 + i) * 50 / data_len)
            sys.stdout.write('\r')
            sys.stdout.write('Computing in progress: [{}{}] {}%'.format("=" * pct_ind, "-" * (50 - pct_ind), pct_ind * 100 / 50) + f' | from {prev_i} to {i}')
            prev_i = i

        # Remainders
        prev_i = i
        i = data_len
        preds = np.squeeze(self.predict(input_x_df.loc[range(prev_i, i), :]))
        pct_ind = 50
        sys.stdout.write('\r')
        sys.stdout.write('Computing in progress: [{}{}] {}%'.format("=" * pct_ind, "-" * (50 - pct_ind), pct_ind * 100 / 50) + f' | from {prev_i} to {i}')
        res_df = res_df.append(pd.DataFrame({'prediction': preds})).reset_index(drop=True)
        print('\nFinished computing all predictions')
        return res_df['prediction']
    
    def delete_endpoint(self):
        try:
            boto3.client('sagemaker').delete_endpoint(EndpointName=self.predictor.endpoint)
            print('Deleted {}'.format(self.predictor.endpoint))
        except:
            print('Already deleted: {}'.format(self.predictor.endpoint))
        self.predictor = None
        return


def evaluate_one_set_result(pred_array, label_array):
    df = pd.DataFrame({'label': label_array, 'pred': pred_array})
    signed_df = np.sign(df[['label', 'pred']])
    correct_sign_pc = signed_df.loc[signed_df['label'] == signed_df['pred'], :].shape[0] / signed_df.shape[0]
    corr = df.corr().iloc[0, 1]
    mse = (df['pred'] - df['label']).pow(2).sum() / df.shape[0]
    return {
        'count': df.shape[0], 
        'sign_accuracy': correct_sign_pc, 
        'correlation': corr, 
        'mse': mse
    }


def evalute_result(train_pred_array, train_label_array, test_pred_array, test_label_array):
    res_dict = {}
    res_dict.update({f'train_{k}': v for k, v in evaluate_one_set_result(train_pred_array, train_label_array).items()})
    res_dict.update({f'test_{k}': v for k, v in evaluate_one_set_result(test_pred_array, test_label_array).items()})
    return res_dict