"""
Perform UPDRS Item 3.4 prediction within ReTap
"""

# import functions and packages
import os
from utils import data_management

def predict_tap_score():

    clf = data_management.load_clf_model()
    print(clf)