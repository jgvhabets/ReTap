"""
Run ReTap from Notebook
"""

# import functions
import os

# import retap functions
from preprocessing import process_raw_acc
from main_scripts.run_retap import main_retap_functionality

# function runned from notebook
def run_ReTap_notebook():
    
    
    print('Notebook run of ReTap')
    
    main_retap_functionality()