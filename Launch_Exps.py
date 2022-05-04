# Raising the Bar in Graph-level Anomaly Detection (GLAD)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import argparse
from config.base import Grid, Config
from evaluation.Experiments import runGraphExperiment
from evaluation.Kfolds_Eval import KFoldEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_OCPool.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='dd')
    return parser.parse_args()

def EndtoEnd_Experiments(config_file, dataset_name):

    model_configurations = Grid(config_file, dataset_name)
    model_configuration = Config(**model_configurations[0])
    dataset =model_configuration.dataset
    result_folder = model_configuration.result_folder+model_configuration.config['model']+'/'+model_configuration.data_name

    exp_class = runGraphExperiment
    risk_assesser = KFoldEval(dataset,result_folder,model_configurations)
    risk_assesser.risk_assessment(exp_class)

if __name__ == "__main__":
    args = get_args()
    config_file = 'config_files/'+args.config_file

    EndtoEnd_Experiments(config_file, args.dataset_name)
