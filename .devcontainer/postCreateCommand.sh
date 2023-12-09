#!/bin/sh
# created by Florian Pfleiderer

# initialise submodule
git submodule update --init --recursive

# download pretrained model
# wget https://thor.robots.ox.ac.uk/cyws-3d/cyws-3d.ckpt.gz

# unzip and delete
# gzip -d cyws-3d.ckpt.gz