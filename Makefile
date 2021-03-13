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
	poetry run isort .
	poetry run black .
	poetry run flake8 . --exit-zero
	poetry run mypy src

## run tests with the default Python
test: lint
	poetry run pytest -vv --cov=lorax -p no:warnings

## check code coverage quickly with the default Python
coverage: clean
	poetry run pytest  -vv --cov=lorax --cov-report html --cov-report term --cov-context=test -p no:warnings
	open -a "Firefox" htmlcov/index.html

## increment the patch version, and tag in git
bumpversion-patch: clean
	poetry version patch

## increment the minor version, and tag in git
bumpversion-minor: clean
	poetry version minor

## increment the major version, and tag in git
bumpversion-major: clean
	poetry version major

## builds source and wheel package
dist: clean
	poetry run poetry build

## install the package to the poetry virtualenv
install: clean
	poetry install

## install the package and all development dependencies to the poetry virtualenv
install-dev: clean
	poetry install --dev

##############################################################################
# Self Documenting Commands                                                  #
##############################################################################
.DEFAULT_GOAL := show-help
# See <https://gist.github.com/klmr/575726c7e05d8780505a> for explanation.
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)";echo;sed -ne"/^## /{h;s/.*//;:d" -e"H;n;s/^## //;td" -e"s/:.*//;G;s/\\n## /---/;s/\\n/ /g;p;}" ${MAKEFILE_LIST}|LC_ALL='C' sort -f|awk -F --- -v n=$$(tput cols) -v i=19 -v a="$$(tput setaf 6)" -v z="$$(tput sgr0)" '{printf"%s%*s%s ",a,-i,$$1,z;m=split($$2,w," ");l=n-i;for(j=1;j<=m;j++){l-=length(w[j])+1;if(l<= 0){l=n-i-length(w[j])-1;printf"\n%*s ",-i," ";}printf"%s ",w[j];}printf"\n";}'
