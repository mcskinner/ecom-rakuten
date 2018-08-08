#!/bin/sh

MODEL=$1; shift
python -m ecom.train $MODEL -- $@
