#!/bin/bash

conda env create -f environment.yml
conda activate pl_jdt
# upgrading pip - https://stackoverflow.com/questions/61365790/error-could-not-build-wheels-for-scipy-which-use-pep-517-and-cannot-be-installe
pip install pip --upgrade
# disabling poetry's experimental new installer - https://github.com/python-poetry/poetry/issues/4210#issuecomment-877778420
poetry config experimental.new-installer false
poetry install
