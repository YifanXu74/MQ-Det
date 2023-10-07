mkdir -p MODEL
mkdir -p DATASET
python -m pip install -r requirements.txt
python setup_glip.py build develop --user
python -m pip install -q -e .
