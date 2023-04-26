"""
Perform UPDRS Item 3.4 prediction within ReTap
"""

# import functions and packages
from os.path import join, exists
from os import makedirs
from numpy import atleast_2d
from pandas import DataFrame
import datetime as dt

from utils import data_management


def predict_tap_score(feats, cfg_filename, verbose=False):
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
        input_X = []
        block_fts = feats[block_name]
        # prepare X vector for prediction (add features in correct order)
        for ft_name in X_ft_list:
            try:
                input_X.append(getattr(block_fts, ft_name))
            except:
                print(f'WARNING: {ft_name} not found for {block_name}')
        
        assert len(input_X) == len(X_ft_list), (
            f'NOT ALL FEATURES FOUND FOR {block_name}, {len(X_ft_list)-len(input_X)} missing'
        )
        input_X = atleast_2d(input_X)
        # predict tap score
        pred_score = clf.predict(input_X)

        if verbose: print(f'Predicted tap score ({block_name}): {pred_score}')
        preds_out.append(pred_score)
    
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