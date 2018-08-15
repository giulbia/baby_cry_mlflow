# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import warnings

from baby_cry_mlflow.rpi_methods import Reader
from baby_cry_mlflow.rpi_methods.feature_engineer import FeatureEngineer
from baby_cry_mlflow.rpi_methods.majority_voter import MajorityVoter

from baby_cry_mlflow.rpi_methods.baby_cry_predictor import BabyCryPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--load_path_model',
                        default='{}/../../../output/model/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../../output/prediction/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--file_name', default='V_2017-04-01+08_04_36=0_13.mp3')

    # Arguments
    args = parser.parse_args()
    load_path_data = os.path.normpath(args.load_path_data)
    load_path_model = os.path.normpath(args.load_path_model)
    file_name = args.file_name
    save_path = os.path.normpath(args.save_path)

    # READ RAW SIGNAL

    # Read signal (first 5 sec)
    file_reader = Reader(os.path.join(load_path_data, file_name))

    play_list = file_reader.read_audio_file()

    # FEATURE ENGINEERING

    # Feature extraction
    engineer = FeatureEngineer()

    play_list_processed = list()

    for signal in play_list:
        tmp = engineer.feature_engineer(signal)
        play_list_processed.append(tmp)

    # MAKE PREDICTION

    # https://stackoverflow.com/questions/41146759/check-sklearn-version-before-loading-model-using-joblib
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)

      with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
          model = pickle.load(fp)

    predictor = BabyCryPredictor(model)

    predictions = list()

    for signal in play_list_processed:
        tmp = predictor.classify(signal)
        predictions.append(tmp)

    # MAJORITY VOTE

    majority_voter = MajorityVoter(predictions)
    majority_vote = majority_voter.vote()

    # SAVE

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'wb') as text_file:
        text_file.write("{}".format(majority_vote))

if __name__ == '__main__':
    main()
