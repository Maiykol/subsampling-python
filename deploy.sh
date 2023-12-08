# build dist
python setup.py sdist

# test on test pypi
# python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/dazer-0.1.22.tar.gz

# upload to pypi
python3 -m twine upload dist/dazer-0.1.22.tar.gz
