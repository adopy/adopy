init:
	pip3 install pipenv
	pipenv install --dev

test:
	pipenv run py.test tests

test-travis:
	pipenv run py.test tests --doctest-modules --cov=adopy
	pipenv run codecov --token $(CODECOV_TOKEN)

lint:
	pipenv run flake8 adopy --format=pylint --statistics --exit-zero
	pipenv run pylint adopy --rcfile=setup.cfg --exit-zero

docs-travis:
	pipenv run travis-sphinx build
	pipenv run travis-sphinx deploy

