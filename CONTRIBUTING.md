# Contributing to FIM-Python

Thank you for your interest in FIM-Python. Any help and contributions are appreciated.



Reporting Bugs
---------------------

Please submit bug reports to the [issue page](https://github.com/thomgrand/fim-python/issues). Make sure that you include all of the following:
- Description of the bug
- Operating system
- Version numbers of
  - Python
  - Numpy
  - Cupy
- Steps to reconstruct the error


Submitting Code
--------------------
FIM-Python uses [pytest](https://docs.pytest.org) to test the code. Pip can take care of installing all necessary packages by listing the extra ``tests``:
```bash
pip install fim-python[gpu,tests]
```
The tests can be run by executing
```bash
python tests/generate_test_data.py #First time only to generate the test examples
python -m pytest tests
```

Before opening a pull request, please make sure that all tests are passing.
In case you only have the CPU version, all tests for the GPU will be skipped. 
The gitlab-runner will also test committed versions of the library, but only on the CPU for the lack of a GPU on the runner.
If you submit new features, please also write tests to ensure functionality of the features.

> **_Note:_**  If you do **not** have a Cupy compatible GPU to test on, please clearly state this in your pull request, so somebody else from the community can test your code with all features enabled.
