### 环境搭建
```shell
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv venv --python 3.12 --seed
$ source .venv/bin/activate
$ pip install notebook
$ pip install jupyter_contrib_nbextensions
$ pip install jupyter_nbextensions_configurator
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```