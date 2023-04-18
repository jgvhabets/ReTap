# guarantee import of all retap modules
import os
import sys

retap_in_sys_path = False

for path in sys.path:
    if path.endswith('retap'):
        retap_in_sys_path = True
        

if not retap_in_sys_path:
    wd = os.getcwd()
    if not wd.endswith('retap'):
        if wd.endswith('ReTap'):
            wd = os.path.join(os.getcwd(), 'src', 'retap')
        while not wd.endswith('retap'):
            wd = os.path.dirname(wd)
        os.chdir(wd)  # set wd to retap

    sys.path.append(os.getcwd())
    print('REPO/src/retap added to SYS PATH')

    