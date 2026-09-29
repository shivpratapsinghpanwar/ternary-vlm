"""Every entry point must at least compile: the unit tests do not import train.py or scripts/, and a syntax error
there otherwise only shows up on Kaggle (stage-2 kernel v11 died this way)."""
import glob
import os
import py_compile

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
FILES = [os.path.join(ROOT, "train.py")] + sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py"))) + \
        sorted(glob.glob(os.path.join(ROOT, "kaggle", "*.py"))) + sorted(glob.glob(os.path.join(ROOT, "ternavlm", "*.py")))


@pytest.mark.parametrize("path", FILES, ids=[os.path.relpath(f, ROOT) for f in FILES])
def test_compiles(path):
    py_compile.compile(path, doraise=True)
