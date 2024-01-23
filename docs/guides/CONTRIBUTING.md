# Contributing

We'd love to accept your patches and contributions to this project. When contributing to the repository, please make sure to first discuss the changes you wish to make via a [github issue](https://github.com/adaptive-intelligent-robotics/QDax/issues).

After the issue is discussed and the solution is determined, you will be invited to fork the repository and create a branch to implement the solution. Once ready to be merged, you can create a [Pull Request](https://github.com/adaptive-intelligent-robotics/QDax/pulls) on github and request to merge into the branch **develop**.

When implementing your contribution, there are just a few guidelines you need to follow.

## Installing Pre-commit hooks

Pre-commits hooks have been configured for this project using the [pre-commit](https://pre-commit.com/) library:

- [black](https://github.com/psf/black) python formatter
- [flake8](https://flake8.pycqa.org/en/latest/) python linter
- [isort](https://pypi.org/project/isort/) sorts imports
- [nbstripout](https://github.com/kynan/nbstripout) strips outputs from notebooks
- [mypy](https://github.com/pre-commit/mirrors-mypy) checks type hints

To get them going on your side, make sure to have python installed, and run the following
commands from the root directory of this repository:

```bash
pip install pre-commit
pre-commit install
```

You can then run the pre-commit hooks on all files as follows:

```bash
pre-commit run --all-files
```

## Coding conventions

Please respect the following conventions to contribute to the code:

- Use hard wrap at 88
- Respect black, isort and flake8 conventions
- Classes' names are Caml case (example: MyClass)
- Functions and variables are in lower case with _ as separator (example: my_function, my_var)
- Names are explicit: avoid mathematical notations, functions' names start with a verb
- Use python typing library: each class and method should be typed (both for inputs and outputs)
- Create custom types if needed
- All classes and functions should have a docstring
- Avoid repeating arguments and returns in docstring (should be explicit with the types) except when it is truly necessary
- A function (or a class) does not take more than 5 arguments, if you need more create a data class
- Avoid dictionaries to pass arguments when possible and prefer dataclasses instead
- Repeat inputs names when calling a function: ex: compute_custom(arg1=arg1, arg2=my_arg2)
- Use list comprehension when it is possible
- Use f strings to add variables in strings: ex: print(f'my var value is {my_var}')

## Commit messages

Please try to follow the conventional [commit standard](https://www.conventionalcommits.org/en/v1.0.0/).

## Merge request and code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
