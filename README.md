# ReTap

## About
ReTap is an open-source tool to generate automated UPDRS finger-tapping predictions and kinematic features describing finger tapping based on index-finger accelerometer data.
A manuscript describing ReTap's intentions, functionality, methodology, and limitations is under review and will follow.

This repo is structured as follows:
```
.
├── LICENSE.txt
├── README.md
├── setup.cfg
├── setup.py
├── pyproject.toml
├── conda_requirements.txt
├── create_conda_env.bat
├── add_env_pip_packages.bat
├── runtime.txt
├── src
│   └── retap
│       ├── main_scripts
│       ├── preprocessing
│       ├── feature_extraction
│       ├── prediction
│       └── utils
├── data
│   └── models
│   └── settings
```

`src` contains the actual modules and functions.
`data` contains the settings and trained models used by the model for tapping score prediction.
other files contain information about the package, the installation, and the use of the functions.


## Quick overview of workflow

### Finding your accelerometer data (DEFINE YOUR LOCAL FOLDER !!)
ReTap will find the (raw) accelerometer files you want to be processed in a FOLDER THAT YOU NEED TO DEFINE. This local folder-location you have to define within `ReTap/data/settings/configs.json`, as variable `raw_acc_folder`.

### Executing ReTap to generate features and predictions
You can run ReTap's functionality either as a python-script directly from the command line, or execute it within a Jupyter Notebook. Both will be explained below.

### Finding the results
ReTap will generate two folders containing the results and the illustrative figures of the feature extraction and the tapping-score-prediction. THESE FOLDER WILL BE CREATED NEXT TO THE FOLDER WITH RAW ACCELEROMETER DATA YOU DEFINED. These folders will be called `retap_results` and `retap_figures`.
```
.
├── YOUR DEFINED FOLDER (in configs.json)
├── retap_figures
│   └── block_detection
├── retap_results
│   └── extracted_tapblocks   (csv files with preprocessed data per detected tapping block)
│   └── features   (json-file with all features on a single-tap-event level, stored per detected tapping block)
│   └── predictions   (csv file with the predicted tapping-score, per detected tapping block)
```

### Checking your results
There are some important steps you need to do, before you can work with the results.

- ReTap is finding blocks of tapping-movement within the (raw) accelerometer data. All results are stored as 'blocks' resulting from a file. These blocks can be visually inspected in `retap_figures/block_detection`. Here you can identify if the block detection was succesful, and you can decide which blocks you will include or discard.


# Installation

### Repository
- GUI: use a git-manager of preference, and clone: https://github.com/jgvhabets/ReTap.git
- Command line:
    - set working directory to desired folder and run: `git clone https://github.com/jgvhabets/ReTap.git`
    - to check initiated remote-repo link, and current branch: `cd ReTap`, `git init`, `git remote -v`, `git branch` (switch to branch main e.g. with `git checkout main`)
- `pip install` from within your DESIGNATED ENVIRONMENT!!:
    - working directory to repo folder, run in Anaconda prompt: `pip install -e .`
    - from any other py-script or notebook: `import retap`

### Environment
(Option 1 preferred)
- 1) conda creation via .yml:
    - working direction should be in repo folder
    - `conda env create -f retap_env.yml -n YOUR_ENV_NAME`
- 2) GUI: Create a python environment with the correct requirements. Either use the GUI of a environments-manager (such as anaconda), and install all dependencies mentioned in the setup.py.
- 3) Command line: you can easily install the required environment from your command line prompt. Note: since some packages are only available via `pip install`, the environment-installation requires 2 commands: one for `conda install`, and one for `pip install`. 
Steps to perform in your (anaconda) prompt:
    - navigate to repo directory, e.g.: `cd Users/USERNAME/Research/ReTap`
    - create environment using batch install: `.\create_conda_env.bat` (confirm Proceed? with `y`)
    - install additional packages to environment (required for pip install packages): `.\add_env_pip_packages.bat`
    - activate conda environment with: `conda activate retap_test`


# User Instruction
- Make sure your environment has the required packages installed, either manually, or by following the instructions above.
- ReTap can be executed directly from the command line, or within a notebook. We will explain both options below.
- In both workflows, ReTap will search for all accelerometer-traces within a predefined folder. THIS FOLDER HAS TO DEFINED WITHIN THE FILE: `ReTap/data/settings/configs.json`.
- Accelerometer traces need to be either Poly5 or csv-files.
- Accelerometer file-names need to contain the following info:
    - their SAMPLING FREQUENCY as UNDERSCORE-FREQ-Hz-UNDERSCORE (e.g. xxx_250Hz_)
    - if file contains only ONE hand, this should be defined e.g. LHand, RHand (see allowed hand codes in class ProcessRawAccData, in process_raw_acc.py)
    - the filenaming will be used for storing the results, so make sure the namings are traceable and differentiable
- MAKE SURE TO CHANGE THE VARIABLE `raw_acc_folder` within configs.json into THE LOCAL FOLDER WHERE YOU STORED THE ACCELEROMETER FILES THAT NEED TO BE PROCESSED.

### Notebook usage
- if `raw_acc_folder` within `ReTap/data/settings/configs.json` is changed succesfully, you can execute ReTap from a notebook, see the example in `src/retap/main_scripts/run_retap.ipynb`

### Command line (py) usage
- if `raw_acc_folder` within `ReTap/data/settings/configs.json` is changed succesfully, you can execute ReTap directly using the script `src/retap/main_scripts/run_retap.py`
- ensure you changed the working directory to the ReTap main repo-directory (`cd Users/USERNAME/FOLDERNAME/ReTap`)
- execute ReTap (on Windows): `python -m src.retap.main_scripts.run_retap` (the `-m` has to be added since the scripts is ran directly from a module) 


# Questions or contribute
Please do not hesitate and reach out in case of any questions, contributions, or what so ever!


# License
This software is available under MIT-LICENSE. Also see the document LICENSE.txt.
