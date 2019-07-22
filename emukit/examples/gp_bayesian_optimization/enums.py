from enum import Enum


class AcquisitionType(Enum):
    EI = 1
    PI = 2
    NLCB = 3
    ES = 4
    SEI = 5


class ModelType(Enum):
    RandomForest = 1
    BayesianNeuralNetwork = 2
