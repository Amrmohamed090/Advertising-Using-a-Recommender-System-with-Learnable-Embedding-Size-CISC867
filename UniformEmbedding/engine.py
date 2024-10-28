import os
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

from models.factorizer import setup_factorizer
from data_loader.data_loader import setup_generator
from utils.evaluate import evaluate_fm 

def setup_args(parser=None):
    """ Set up arguments for the Engine """
    if parser is None:
        parser = ArgumentParser()
    data = parser.add_argument_group('Data')
    engine = parser.add_argument_group('Engine Arguments')
    factorize = parser.add_argument_group('Factorizer Arguments')
    matrix_factorize = parser.add_argument_group('MF Arguments')
    log = parser.add_argument_group('Tensorboard Arguments')

    engine.add_argument('--alias', default='experiment', help='Name for the experiment')
    engine.add_argument('--seed', default='42')

    data.add_argument('--data-type', default='ml1m', help='type of the dataset')
    data.add_argument('--data-path', default='./data/{data_type}/')
    data.add_argument('--batch-size-train', default=1)
    data.add_argument('--batch-size-valid', default=1)
    data.add_argument('--batch-size-test', default=1)

    engine.add_argument('--max-steps', default=1e5)
    engine.add_argument('--use-cuda', default=True)
    engine.add_argument('--device-id', default=0, help='Training Devices')
    engine.add_argument('--early-stop', type=int, default=10, help="Early stopping steps")

    factorize.add_argument('--factorizer', default='fm', help='Type of the Factorization Model')
    factorize.add_argument('--latent-dim', default=8)
    

    type_opt = 'fm'
    matrix_factorize.add_argument('--{}-optimizer'.format(type_opt), default='sgd')
    matrix_factorize.add_argument('--{}-lr'.format(type_opt), default=1e-3)

    log.add_argument('--log-interval', default=1)
    log.add_argument('--tensorboard', default='./tmp/runs')
    log.add_argument('--display_interval', default=100)
    return parser

class Engine(object):
    """Engine wrapping the training & evaluation of standard FM"""
    _global_writer = None  # Class variable for the global writer
    _param_step = 0  # Class variable to track parameter steps

    def __init__(self, opt):
        self._opt = opt
        self._opt['data_path'] = self._opt['data_path'].format(data_type=self._opt['data_type'])
        self._sampler = setup_generator(opt)
        self._opt['field_dims'] = self._sampler.field_dims
        self._factorizer = setup_factorizer(opt)

        if Engine._global_writer is None:
            Engine._global_writer = SummaryWriter(log_dir='{}/parameter_comparison'.format(self._opt['tensorboard']))
        
        # Individual run writer
        self._writer = SummaryWriter(log_dir='{}/{}'.format(self._opt['tensorboard'], opt['alias']))
        self._n_parameters = opt['field_dims'] * opt['latent_dim']
        print(f"number of embedding parameters: {self._n_parameters}")

    def train_an_episode(self, max_steps):
        """Standard training loop for a Factorization Machine"""
        print(f"Training episode for {max_steps} steps.")
        
        log_interval = self._opt.get('log_interval')
        display_interval = self._opt.get('display_interval')
        best_test_result = {
            "AUC": [0, 0], 
            "LogLoss": [np.inf, 0], 
            "n_parameters": self._n_parameters
        }

        # Initialize early stopping variables
        best_train_loss = float('inf')
        best_valid_loss = float('inf')
        no_improvement_steps = 0  # Counter for early stopping

        epoch_start = datetime.now()
        for step_idx in range(int(max_steps)):
            # Perform a training step
            train_mf_loss = self._factorizer.update(self._sampler)

            # Logging
            if step_idx % log_interval == 0:
                self._writer.add_scalar('train/step_wise/mf_loss', train_mf_loss, step_idx)

            if step_idx % display_interval == 0:
                print(f'[Step {step_idx}] MF Loss: {train_mf_loss:.4f}')

            # Evaluate periodically
            if step_idx % self._sampler.num_batches_train == 0:
                logloss, auc = evaluate_fm(self._factorizer, self._sampler, self._opt['use_cuda'])
                self._writer.add_scalar('test/epoch_wise/metron_auc', auc, step_idx)
                self._writer.add_scalar('test/epoch_wise/metron_logloss', logloss, step_idx)

                if logloss < best_test_result['LogLoss'][0]:
                    best_test_result['LogLoss'] = [logloss, step_idx]
                if auc > best_test_result['AUC'][0]:
                    best_test_result['AUC'] = [auc, step_idx]

                print(f"[Evaluation] AUC: {auc:.4f} | LogLoss: {logloss:.4f}")

                # Check if there is improvement in either training or validation loss
                if logloss < best_valid_loss:
                    best_valid_loss = min(logloss, best_valid_loss)
                    no_improvement_steps = 0  # Reset counter if improvement
                    print("Improvement found in validation loss.")
                else:
                    no_improvement_steps += 1  # Increment counter if no improvement

            # Early stopping based on lack of improvement
            if no_improvement_steps >= self._opt['early_stop']:
                print("Early stop triggered due to no improvement in training/validation loss.")
                print("Best test results:", best_test_result)
                self.train_finish(best_test_result)
                return best_test_result

        self.train_finish(best_test_result)
        print(f"max_step reached, best result is: {best_test_result}")
        
        return best_test_result

    def train(self):
        return self.train_an_episode(self._opt['max_steps']) # returns best test result
        
    def train_finish(self, best_test_result):
        Engine._global_writer.add_scalar(
            'parameter_comparison/best_auc_vs_params', 
            best_test_result['AUC'][0],
            Engine._param_step
        )
        Engine._param_step += 1  # Increment the step counter
        Engine._global_writer.flush()

    def __del__(self):
        if hasattr(self, '_writer'):
            self._writer.close()
        if Engine._global_writer is not None:  # Close global writer if exists
            Engine._global_writer.close()



if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    opt = vars(opt)  # Convert to dictionary
    engine = Engine(opt)
    engine.train()
