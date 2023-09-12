"""
Main script to run ReTap from command line
"""

# import packages and functions
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src', 'retap'))

# import retap's functions
from preprocessing import process_raw_acc
from feature_extraction import feat_extraction_classes as ftClasses
from feature_extraction.extract_features import run_ft_extraction
from prediction import predict_score

def main_retap_functionality(cfg_filename='configs_adbs.json',
                             single_file=None,
                             verbose=False):
    """
    Function that runs ReTap algorithm parts, will
    be ran both in command-line as in notebook use.
    """
    # Part 1: load raw-ACC and detect-active blocks
    rawAcc = process_raw_acc.ProcessRawAccData(
        cfg_filename=cfg_filename,
        use_single_file=single_file,
        verbose=verbose
    )

    # Part 2 and 3: detect single taps and feature extraction
    fts = run_ft_extraction(acc_block_names=rawAcc.current_trace_list,
                            cfg_filename=cfg_filename,
                            verbose=verbose)

    # Part 4: create predicted UPDRS Item 3.4 score
    predict_score.predict_tap_score(feats=fts, cfg_filename=cfg_filename,
                                    verbose=verbose)

    return 'retap ran succesfully'



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
    # if single filename given
    if len(sys.argv) == 2:
        file_sel = sys.argv[1]
        print(f'ReTap performed on file: {file_sel}')
        main_retap_functionality(single_file=file_sel)
    
    
    # otherwise use all files in designated folder
    else:
        # call function with all ReTap functionality
        print('PROCESS ALL AVAILABLE DATA')
        main_retap_functionality()
    

