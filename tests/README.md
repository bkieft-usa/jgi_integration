# Running Tests

From the repository root, run:

```sh
python -m pytest -q
```

_Note_: If you are getting module import errors, ensure you are at the repo root and not inside a subdir.

The tests use synthetic in-memory data and do not require project input files or a running Docker container. Use a Python environment with `pytest`, `numpy`, `pandas`, and `scipy` installed.

For verbose test names and results, run:

```sh
python -m pytest -v
```
