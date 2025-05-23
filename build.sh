#!/bin/bash

set -e

rm -rf build/ dist/ *.egg-info/
python -m build
ls -la dist/

echo "To upload: twine upload dist/*"
echo "TestPyPI: twine upload --repository testpypi dist/*"
echo "Now installing built package"
pip install -e .