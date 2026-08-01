#!/bin/bash

source $CONDA_SHELL

mamba create -p env/stitch python==3.12 -y
mamba activate env/stitch

pip install bezier[full] ipykernel python_tsp
pip install -e .
pip install -e ~/Software/mrich
python -c "import mrich; mrich.patch_rich_jupyter_margins()"
