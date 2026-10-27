"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pde.tools.mpi import mpi_allreduce, mpi_recv, mpi_send, rank, size


def test_mpi_error_exit_status(tmp_path):
    """An error inside an MPI solver must fail the launcher as well."""
    if size > 1:
        pytest.skip("This test launches its own MPI job")
    pytest.importorskip("mpi4py")
    pytest.importorskip("numba_mpi")
    launcher = shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("MPI launcher is not available")

    script = tmp_path / "mpi_error.py"
    script.write_text(
        textwrap.dedent("""\
            from mpi4py import MPI
            from pde import PDE, ScalarField, UnitGrid

            assert MPI.COMM_WORLD.size == 2
            print(f"ready:{MPI.COMM_WORLD.rank}", flush=True)
            MPI.COMM_WORLD.Barrier()
            PDE({"c": "undefined_rhs"}).solve(
                ScalarField(UnitGrid([4])), solver="explicit_mpi", t_range=1
            )
            """),
        encoding="utf-8",
    )
    command = [launcher, "-n", "2"]
    if sys.platform == "darwin":
        command += ["-host", "localhost:2"]
    command += [sys.executable, str(script)]
    package_path = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=120
    )
    output = result.stdout + result.stderr
    assert "ready:0" in result.stdout, output
    assert "ready:1" in result.stdout, output
    assert "undefined_rhs" in result.stderr, output
    assert result.returncode != 0, output


@pytest.mark.multiprocessing
def test_send_recv():
    """Test basic send and receive."""
    if size == 1:
        pytest.skip("Run without multiprocessing")

    data = np.arange(5)
    if rank == 0:
        out = np.empty_like(data)
        mpi_recv(out, 1, 1)
        np.testing.assert_allclose(out, data)
    elif rank == 1:
        mpi_send(data, 0, 1)


@pytest.mark.multiprocessing
@pytest.mark.parametrize("operator", ["MAX", "MIN", "SUM"])
def test_allreduce(operator, rng):
    """Test MPI allreduce function."""
    data = rng.uniform(size=size)
    result = mpi_allreduce(data[rank], operator=operator)

    if operator == "MAX":
        assert result == data.max()
    elif operator == "MIN":
        assert result == data.min()
    elif operator == "SUM":
        assert result == data.sum()
    else:
        raise NotImplementedError


@pytest.mark.multiprocessing
@pytest.mark.parametrize("operator", ["MAX", "MIN", "SUM"])
def test_allreduce_array(operator, rng):
    """Test MPI allreduce function."""
    data = rng.uniform(size=(size, 3))
    result = mpi_allreduce(data[rank], operator=operator)

    if operator == "MAX":
        np.testing.assert_allclose(result, data.max(axis=0))
    elif operator == "MIN":
        np.testing.assert_allclose(result, data.min(axis=0))
    elif operator == "SUM":
        np.testing.assert_allclose(result, data.sum(axis=0))
    else:
        raise NotImplementedError
