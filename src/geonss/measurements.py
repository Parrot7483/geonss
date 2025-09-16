"""
This module provides functionality for handling and processing GNSS measurements.
"""

from enum import Enum

class MeasurementType(Enum):
    """
    Class to represent a measurement type.
    Pseudo range, Phase, SNR,
    """
    PSEUDO_RANGE = "C"
    PHASE = "L"
    SNR = "S"

    def __str__(self):
        if self == MeasurementType.PSEUDO_RANGE:
            return "Pseudo Range"
        if self == MeasurementType.PHASE:
            return "Phase"
        if self == MeasurementType.SNR:
            return "SNR"

        return "Unknown"

    @staticmethod
    def get_type(meas: str) -> 'MeasurementType':
        """
        Get the measurement type from the RINEX identifier.

        Args:
            meas: The RINEX identifier (e.g. C1C, L1C, S1C)

        Returns:
            MeasurementType: The measurement type
        """
        if meas[0] == "C":
            return MeasurementType.PSEUDO_RANGE

        if meas[0] == "L":
            return MeasurementType.PHASE

        if meas[0] == "S":
            return MeasurementType.SNR

        raise ValueError(f"Unknown measurement type for {meas}")


def get_frequency(meas: str) -> float:
    """
    Get the frequency of a pseudo range measurement.
    """
    if meas[0] == "C":
        # Get the frequency from the measurement type
        if meas[1] == "1":
            return 1575.42

        if meas[1] == "2":
            return 1227.60

        if meas[1] == "5":
            return 1176.45

        raise ValueError(f"Unknown frequency for {meas}")

    raise ValueError(f"Unknown measurement type for {meas}")
