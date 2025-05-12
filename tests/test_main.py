import subprocess
import os
import tempfile
from tests.util import path_test_file
import georinex as gr
import numpy as np


def test_main():
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output = os.path.join(temp_dir, "main-test.sp3")

        navigation = path_test_file("BRDC00IGS_R_20250980000_01D_MN.rnx")
        observation = path_test_file("WTZR00DEU_R_20250980000_01D_30S_MO.crx")

        # Construct the command
        cmd = [
            "geonss",
            "-n", navigation,
            "--time-limit", "2025-04-08T12:00:00,2025-04-08T013:00:00",
            "--disable-gps",
            observation,
            "--output", output
        ]

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check that the process completed successfully
        assert result.returncode == 0, f"Process failed with output: {result.stderr}"

        # Check that the output file was created
        assert os.path.exists(output), "Output file was not created"

        try:
            result = gr.load(output)
        except Exception as e:
            raise AssertionError(f"Failed to load output file: {e}")

        mean_position = result.position.squeeze().mean(dim="time").values
        real_position = np.array([4075.5808863, 931.8535784, 4801.5679707])

        # Check that the mean position is close to the expected value
        assert np.allclose(mean_position, real_position, atol=1e-6), "Mean position does not match expected value"

        # Check that the mean position is within 3 m of the expected value
        assert np.linalg.norm(mean_position - real_position) * 1000 < 3, "Mean position does not match expected value"
