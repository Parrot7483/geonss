# noinspection SpellCheckingInspection
"""
GNSS Constellation Management Module

This module provides functionality for working with GNSS (Global Navigation Satellite System) constellations.
It includes:
- Constellation enum for identifying different observable systems
- Functions for mapping observable IDs to their respective constellations
- Filtering capabilities for processing datasets based on constellation types

The module supports major GNSS systems including:
- GPS (Global Positioning System)
- GLONASS (GLObal Navigation Satellite System)
- Galileo (European GNSS)
- BeiDou (Chinese Navigation Satellite System)
- QZSS (Quasi-Zenith Satellite System)
- SBAS (Satellite Based Augmentation System)
- IRNSS (Indian Regional Navigation Satellite System)
"""

from enum import Enum
from typing import Union

import xarray as xr


# noinspection SpellCheckingInspection
class Constellation(Enum):
    """
    Enum representing GNSS constellations.
    """
    GPS = "gps"
    GLONASS = "glonass"
    GALILEO = "galileo"
    BEIDOU = "beidou"
    QZSS = "qzss"
    SBAS = "sbas"
    IRNSS = "irnss"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


def get_common_satellites(dataset_a: xr.Dataset, dataset_b: xr.Dataset) -> list[str]:
    """
    Find observable IDs that exist in both datasets.

    Args:
        dataset_a: First dataset containing 'sv' coordinate with observable IDs
        dataset_b: Second dataset containing 'sv' coordinate with observable IDs

    Returns:
        list[str]: List of observable IDs present in both datasets
    """
    sats_a = dataset_a['sv'].values.tolist()
    sats_b = dataset_b['sv'].values.tolist()
    return list(set(sats_a) & set(sats_b))


def get_constellation(satellite_id: str) -> Constellation:
    """
    Maps a observable identifier to its constellation.

    Args:
        satellite_id: Satellite ID in format [A-Z]\\d{2} (e.g. G01, R24, E11)

    Returns:
        The constellation enum for the observable
    """
    if not isinstance(satellite_id, str) or len(satellite_id) != 3 or not satellite_id[1:].isdigit():
        return Constellation.UNKNOWN

    constellation_map = {
        'G': Constellation.GPS,
        'R': Constellation.GLONASS,
        'E': Constellation.GALILEO,
        'C': Constellation.BEIDOU,
        'B': Constellation.BEIDOU,
        'J': Constellation.QZSS,
        'S': Constellation.SBAS,
        'I': Constellation.IRNSS
    }

    return constellation_map.get(
        satellite_id[0].upper(), Constellation.UNKNOWN)


# noinspection SpellCheckingInspection
def select_constellations(
        df: Union[xr.Dataset, xr.DataArray],
        galileo: bool = False,
        gps: bool = False,
        beidou: bool = False,
        glonass: bool = False,
        sbas: bool = False,
        underscores: bool = False
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Select which GNSS constellations to keep in the dataset.

    Args:
        df : xarray.Dataset or xarray.DataArray
            Input data containing observable identifiers in a 'sv' coordinate/variable
        galileo : bool, optional
            If True, keep Galileo satellites (prefix 'E'), by default False
        gps : bool, optional
            If True, keep GPS satellites (prefix 'G'), by default False
        beidou : bool, optional
            If True, keep BeiDou satellites (prefixes 'C' and 'B'), by default False
        glonass : bool, optional
            If True, keep GLONASS satellites (prefix 'R'), by default False
        sbas : bool, optional
            If True, keep SBAS satellites (prefix 'S'), by default False
        underscores : bool, optional
            If True, remove entries containing underscores in observable ID, by default False

    Returns:
        xarray.Dataset or xarray.DataArray
            Filtered data containing only selected constellations
    """
    constellation_filters = {
        Constellation.GALILEO: galileo,
        Constellation.GPS: gps,
        Constellation.BEIDOU: beidou,
        Constellation.GLONASS: glonass,
        Constellation.SBAS: sbas
    }

    def keep_sv(sv_id: str) -> bool:
        if '_' in sv_id and underscores:
            return False
        constellation = get_constellation(sv_id)
        return constellation_filters.get(constellation, True)

    mask = [keep_sv(sv) for sv in df.sv.values]
    return df.sel(sv=df.sv[mask])


def select_satellites(
        df: Union[xr.Dataset, xr.DataArray],
        sv_list: list[str]
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Select specific satellites to keep in the dataset.

    Args:
        df : xarray.Dataset or xarray.DataArray
            Input data containing observable identifiers in a 'sv' coordinate/variable
        sv_list : list[str]
            List of observable IDs to keep (e.g., ['G01', 'R24', 'E11'])

    Returns:
        xarray.Dataset or xarray.DataArray
            Data containing only specified satellites
    """
    sv_set = {sv.upper() for sv in sv_list}
    return df.sel(sv=[s for s in df.sv.values if s.upper() in sv_set])
