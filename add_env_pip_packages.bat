@echo off
conda run -n retap_test pip install --proxy=http://proxy.charite.de:8080 mne==1.2.1 pingouin==0.5.3
echo Succesfully added mne and pingouin packages!