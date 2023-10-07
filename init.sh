mkdir -p MODEL
mkdir -p DATASET
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup_glip.py build develop --user
python -m pip install -q -e .
