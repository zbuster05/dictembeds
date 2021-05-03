#!/bin/bash

BASEPATH=$1

pushd "$BASEPATH"
python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
popd

