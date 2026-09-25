#!/usr/bin/env bash
# Public COVID-19 Image Data Collection (Cohen et al., 2020)
set -e
mkdir -p data/raw
[ -d data/raw/covid-chestxray-dataset ] || git clone --depth 1 https://github.com/ieee8023/covid-chestxray-dataset data/raw/covid-chestxray-dataset
