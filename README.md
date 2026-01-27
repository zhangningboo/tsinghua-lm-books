### 环境搭建
```shell
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv venv --python 3.12 --seed
$ source .venv/bin/activate
$ uv pip install notebook
$ uv pip install jupyter_contrib_nbextensions
$ uv pip install jupyter_nbextensions_configurator
$ uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
$ uv pip install transformers
$ uv pip install modelscope
$ uv pip install scikit-learn
```