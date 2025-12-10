#!/bin/bash

set -euo pipefail

python -m src.train.train_dyck "$@"

