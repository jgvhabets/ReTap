"""
Main script to run ReTap from command line
"""

# import packages and functions
import sys
import os

# import retap's functions
from preprocessing import process_raw_acc
from feature_extraction import feat_extraction_classes as ftClasses
from feature_extraction.run_feat_extract import run_ft_extraction

def main_retap_functionality(cfg_filename='config_jh.json'):
    """
    Function that runs ReTap algorithm parts, will
    be ran both in command-line as in notebook use.
    """
    # Part 1: load raw-ACC and detect-active blocks
    rawAcc = process_raw_acc.ProcessRawAccData(
        cfg_filename=cfg_filename,
    )

    # Part 2 and 3: detect single taps and feature extraction
    run_ft_extraction(sel_acc_blocks=rawAcc.current_trace_list,
                      cfg_filename=cfg_filename,)

    # Part 4: create predicted UPDRS Item 3.4 score


    return 'retap functionality'

# function runs when directly called from command line
if __name__ == '__main__':
    """
    This functions call the function above 'main_retap_functionality'
    and ensures that ReTap's functionality from the command-line
    is identical to it's functionality from a notebook
    (run_retap_notebook.ipynb).

    To be called on WIN (e.g. from ReTap working directory):
        python -m src.retap.main_scripts.run_retap
    """
    # call function with all ReTap functionality
    main_retap_functionality()
    

