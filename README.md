# ReTap
ReTap is an open-source tool to generate automated UPDRS finger-tapping predictions based on index-finger accelerometer data

# Installation

### Repo:
- GUI: use a git-manager of preference, and clone: https://github.com/jgvhabets/ReTap.git
- Command line:
    - set working directory to desired folder and run: `git clone https://github.com/jgvhabets/ReTap.git`
    - check initiated remote-repo link, and current branch: `cd ReTap`, `git init`, `git remote -v`, `git branch` (switch to branch main e.g. with `git checkout main`) 

### Environment
- GUI: Create a python environment with the correct requirements. Either use the GUI of a environments-manager (such as anaconda), and install all dependencies mentioned in the setup.py.
- Command line: run in the (anaconda) prompt:
    - navigate to repo directory, e.g.: `cd Users/USERNAME/Research/ReTap`
    - create environment using batch install: `.\create_conda_env.bat` (confirm Proceed? with `y`)
    - install additional packages to environment (required for pip install packages): `.\add_env_pip_packages.bat`
    - activate conda environment with: `conda activate retap_test`

# License
This software is available under MIT-LICENSE. Also see the document LICENSE.
