# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle

import numpy as np

from baby_cry_mlflow.pc_methods.train_classifier import TrainClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../../../output/dataset/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../../output/model/'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)

    # TRAIN MODEL

    X = np.load(os.path.join(load_path, 'dataset.npy'))
    y = np.load(os.path.join(load_path, 'labels.npy'))

    train_classifier = TrainClassifier(X, y)
    performance, parameters, best_estimator = train_classifier.train()

    # SAVE
    #
    #  Save performances
    # with open(os.path.join(save_path, 'performance.json'), 'w') as fp:
    #     json.dump(performance, fp)
    #
    # # Save parameters
    # with open(os.path.join(save_path, 'parameters.json'), 'w') as fp:
    #     json.dump(parameters, fp)
    #
    # # Save model
    # with open(os.path.join(save_path, 'model.pkl'), 'wb') as fp:
    #     pickle.dump(best_estimator, fp)

if __name__ == '__main__':
    main()
