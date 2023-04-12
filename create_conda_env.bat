@echo off

setlocal

set ENV_NAME=retap_test
set REQUIREMENTS=conda_requirements.txt

echo Creating environment %ENV_NAME% ...

conda create --name %ENV_NAME% --file %REQUIREMENTS%

echo Environment %ENV_NAME% created successfully!

endlocal
