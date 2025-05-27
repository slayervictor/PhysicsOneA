#!/bin/bash

set -e

python importhandler.py
rm -rf build/ dist/ *.egg-info/
python -m build
ls -la dist/

pip install -e .

echo "To upload: twine upload dist/*"
echo "TestPyPI: twine upload --repository testpypi dist/*"
echo "Now installing built package"
