"""
Perform UPDRS Item 3.4 prediction within ReTap
"""

# import functions and packages
from os.path import join, exists
from os import makedirs
from numpy import atleast_2d, float64, isnan, nan
from pandas import DataFrame
import datetime as dt

from utils import data_management


def predict_tap_score(
    feats: dict, cfg_filename: str, min_n_taps: int = 9,
    verbose: bool = False
):
    """
    Function to generate predictions for included
    tapping blocks

    Input:
        - feats (dict): containing feat-class per
            trace
        - cfg_filename (str): settings json filename
        - min_n_taps (int): if a trace contains less
            detected taps than this number, the trace
            will be predicted as a 3
    """
    # load settings for model name and feature list
    settings = data_management.read_cfg_file(cfg_filename=cfg_filename)
    # load classifier, saved as pickle
    clf = data_management.load_clf_model(
        clf_fname=settings['clf_model_filename'])
    # define feature list for model input
    X_ft_list = settings['clf_input_features']
    
    # prediction list to store
    preds_out = []
    # define path to store
    # find available tapping block files
    paths = data_management.get_directories_from_cfg(cfg_filename=cfg_filename)
    pred_path = join(paths['results'], 'predictions')
    if not exists(pred_path): makedirs(pred_path)

    # perform tap-score prediction for every tapping-block included
    for block_name in feats.keys():

        block_fts = feats[block_name]  # take fts-class for block

        # check number of taps present
        class_on_n_taps = classify_based_on_nTaps(
            block_taps=block_fts.tap_lists, block_feats=block_fts,
            min_n_taps=min_n_taps
        )
        if class_on_n_taps:
            preds_out.append(class_on_n_taps)
            if verbose: print(f'Predicted tap score ({block_name}):'
                              f' {class_on_n_taps} (based on n-taps)')
            continue
        
        # predict block using classification
        input_X = []
        
        # prepare X vector for prediction (add features in correct order)
        for ft_name in X_ft_list:
            try:
                input_X.append(getattr(block_fts, ft_name))
            except:
                print(f'WARNING: {ft_name} not found for {block_name}')
        
        assert len(input_X) == len(X_ft_list), (
            f'NOT ALL FEATURES FOUND FOR {block_name}, {len(X_ft_list)-len(input_X)} missing'
        )
        input_X = atleast_2d(input_X).astype(float64)
        if isnan(input_X).any(): 
            print(f'{block_name} features contained NaNs')
            preds_out.append(nan)
            continue
        # predict tap score
        pred_score = clf.predict(input_X)[0]
                    
        preds_out.append(pred_score.astype(int))

        if verbose: print(f'Predicted tap score ({block_name}): {pred_score}')
    
    # create filename to store predictions,
    # includes date and does not overwrite existing files
    dd = str(dt.date.today().day).zfill(2)
    mm = str(dt.date.today().month).zfill(2)
    yyyy = dt.date.today().year
    today = f'{yyyy}{mm}{dd}'
    run = 1
    filename = f'{today}_predicted_scores_run{run}.csv'
    while exists(join(pred_path, filename)):
        run += 1
        filename = f'{today}_predicted_scores_run{run}.csv'
    
    # save results as table
    preds = DataFrame(data=preds_out, index=list(feats.keys()),
                      columns=['predicted_tap_score'])
    
    preds.to_csv(join(pred_path, filename), sep=',')

    return 'predictions completed and stored successfully'


def classify_based_on_nTaps(
    block_taps, block_feats,
    min_n_taps: int = 9, score_to_predict: int = 3,
):
    """
    Classify traces based on a too small number of
    taps detected

    Input:
        - max_n_taps: threshold of taps present
        - ftClass: features used
        - score_to_set: score to be classified with
        - in_cv: performed in cross-validation, important
            for true label handling
    
    Returns:
        - CLASSIFY_BLOCK: if block does not meet criteria
            for prediction based on n-taps: False, if
            criteria are met: score_to_predict
    """
    CLASSIFY_BLOCK = False

    n_taps_detected = len(block_taps)

    if n_taps_detected < min_n_taps:
        CLASSIFY_BLOCK = score_to_predict

        # check escape for traces with few taps, but large amplitudes
        if sum(block_feats.raise_velocity) > 100:
            
            CLASSIFY_BLOCK = False

    return CLASSIFY_BLOCK