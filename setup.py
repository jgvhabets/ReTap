from setuptools import setup, find_packages

setup(
    name='ReTap',
    version='0.1.0',
    author='Jeroen Habets',
    author_email='jgvhabets@gmail.com',
    description='ReTap is an open-source tool to generate automated UPDRS finger-tapping predictions based on index-finger accelerometer data',
    long_description='',
    url='https://github.com/jgvhabets/ReTap',
    packages=find_packages(where="src"),
    install_requires=['jupyter', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'h5py', 'mne', 'openpyxl', 'pengouin'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Neurological, Movement Disorders, Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires = ">=3.9.13",
)
