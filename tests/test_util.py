import numpy as np
from geonss.parsing import load_parallel
from georinex import load
from datetime import datetime
import logging
from tests.util import path_test_file

logging.basicConfig(level=logging.DEBUG)

def test_load_parallel1():
    obs = path_test_file("GEOP091P.25o")

    compare = load(obs)

    combined = load_parallel(obs)

    assert np.all(combined == compare)

def test_load_parallel2():
    obs = path_test_file("s6an0010.24o")
    start = datetime.fromisoformat("2024-01-01T03:28:00")
    end = datetime.fromisoformat("2024-01-01T03:30:00")

    compare = load(obs, tlim=(start, end))

    combined = load_parallel(obs, tlim=(start, end), processes=4)

    assert np.all(combined == compare)
