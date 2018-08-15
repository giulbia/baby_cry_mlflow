# -*- coding: utf-8 -*-

import argparse
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import mlflow
import mlflow.sklearn


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

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=0,
                                                        stratify=y)

    pipeline = Pipeline([
        ('scl', StandardScaler()),
        ('clf', SVC(probability=True))
    ])

    # GridSearch
    param_grid = [{'clf__kernel': ['linear', 'rbf'],
                   'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'clf__gamma': np.logspace(-2, 2, 5),
                   }]

    estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

    with mlflow.start_run():

        model = estimator.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='macro'))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='macro'))
        mlflow.log_metric("f1", f1_score(y_test, y_pred, average='macro'))


if __name__ == '__main__':
    main()
