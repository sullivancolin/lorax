.PHONY: clean clean-test clean-pyc clean-build

## remove all build, test, coverage and Python artifacts
clean: clean-build clean-pyc clean-test

## remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg' -exec rm -f {} +

## remove Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -rf .mypy_cache/

## remove test and coverage artifacts
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .cache/
	rm -fr .pytest_cache
	rm -f coverage.xml

## check style with flake8, mypy, black
lint: clean
	pipenv run isort .
	pipenv run black .
	pipenv run flake8 . --exit-zero
	pipenv run mypy src

## run tests with the default Python
test: lint
	pipenv run pytest -vv --cov=src

## check code coverage quickly with the default Python
coverage: clean
	pipenv run pytest  -vv --cov=src --cov-report html --cov-report term --cov-context=test
	open -a "Firefox" htmlcov/index.html

## increment the patch version, and tag in git
bumpversion-patch: clean
	pipenv run bumpversion --verbose patch

## increment the minor version, and tag in git
bumpversion-minor: clean
	pipenv run bumpversion --verbose minor

## increment the major version, and tag in git
bumpversion-major: clean
	pipenv run bumpversion --verbose major

## builds source and wheel package
dist: clean
	pipenv run python setup.py sdist bdist_wheel

## install the package to the pipenv virtualenv
install: clean
	pipenv install

## install the package and all development dependencies to the pipenv virtualenv
install-dev: clean
	pipenv install --dev

##############################################################################
# Self Documenting Commands                                                  #
##############################################################################
.DEFAULT_GOAL := show-help
# See <https://gist.github.com/klmr/575726c7e05d8780505a> for explanation.
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)";echo;sed -ne"/^## /{h;s/.*//;:d" -e"H;n;s/^## //;td" -e"s/:.*//;G;s/\\n## /---/;s/\\n/ /g;p;}" ${MAKEFILE_LIST}|LC_ALL='C' sort -f|awk -F --- -v n=$$(tput cols) -v i=19 -v a="$$(tput setaf 6)" -v z="$$(tput sgr0)" '{printf"%s%*s%s ",a,-i,$$1,z;m=split($$2,w," ");l=n-i;for(j=1;j<=m;j++){l-=length(w[j])+1;if(l<= 0){l=n-i-length(w[j])-1;printf"\n%*s ",-i," ";}printf"%s ",w[j];}printf"\n";}'
