# -*- coding: utf-8 -*-

import argparse
import re
import os
import numpy as np
import mlflow

from baby_cry_mlflow.utils import Reader
from baby_cry_mlflow.utils.feature_engineer import FeatureEngineer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../../data'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../../output/dataset/'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)

    # READ FILES IN SUB-FOLDERS of load_path and FEATURE ENGINEERING

    with mlflow.start_run():

        # list load_path sub-folders
        regex = re.compile(r'^[0-9]')
        directory_list = [i for i in os.listdir(load_path) if regex.search(i)]

        # initialize empty array for features
        X = np.empty([1, 18])

        # initialise empty array for labels
        y = []

        # iteration on sub-folders
        for directory in directory_list:

            # Instantiate FeatureEngineer
            feature_engineer = FeatureEngineer(label=directory)

            file_list = os.listdir(os.path.join(load_path, directory))

            # iteration on audio files in each sub-folder
            for audio_file in file_list:
                file_reader = Reader(os.path.join(load_path, directory, audio_file))
                data, sample_rate = file_reader.read_audio_file()
                avg_features, label = feature_engineer.feature_engineer(audio_data=data)

                X = np.concatenate((X, avg_features), axis=0)
                y.append(label)

        # X.shape is (401, 18) as I'm not using indexing. First line is made of zeros and is to be removed
        X = X[1:, :]

        # Save to numpy binary format
        np.save(os.path.join(save_path, 'dataset.npy'), X)
        np.save(os.path.join(save_path, 'labels.npy'), y)

if __name__ == '__main__':
    main()
