# MAGIC Global Positioning L1B Processor

A single-point positioning system for processing satellite data to compute positions, velocities, and clock biases. Developed as part of an Interdisciplinary Project in an Application Subject for the Master in Computer Science at the [Ingenieurinstitut für Astronomische und Physikalische Geodäsie](https://www.asg.ed.tum.de/en/iapg/startseite/) at [Technical University of Munich](https://www.tum.de/).

## Introduction

The MAGIC Global Positioning L1B Processor is part of the Mass-Change and Geosciences International Constellation (MAGIC) mission by NASA and ESA. The mission aims to measure Earth's gravity changes with improved accuracy and resolution. This system processes satellite data to compute single-point positions, applying iteratively reweighted least squares to refine the receiver's position, velocity, and clock bias in an Earth-Centered, Earth-Fixed (ECEF) frame.

## Features

- Dual-frequency signals
- Dual constellation (GPS and Galileo)
- Input formats: RINEX, ANTEX, SP3
- Output format: SP3
- Object-oriented LLA and ECEF position handling
- Parallel processing and caching for handling large datasets

## Installation

The software can be installed using pip:

```bash
$ pip install git+https://github.com/Parrot7483/gnss.git
```

## Usage

The software can be used from the command line or as a Python library.

### Command Line Interface (CLI)

```bash
$ geonss --help
usage: geonss [-h] [-i ID] [-n NAVIGATION] [-s SP3] [-a ANTEX] [-o OUTPUT] [-t TLIM] [-v]
              [--disable-galileo] [--disable-gps] [--disable-signal-travel-time-correction]
              [--disable-earth-rotation-correction] [--disable-tropospheric-correction]
              [--disable-elevation-weighting] [--disable-snr-weighting]
              observation

A single point positioning program for GNSS data. Supports Galileo and GPS. Measurements must be dual frequency.

positional arguments:
  observation           Path to RINEX observation file. (required path).

options:
  -h, --help            show this help message and exit
  -i ID, --identifier ID
                        Receiver identifier (default: L01) (optional string).
  -n NAVIGATION, --navigation NAVIGATION
                        Path to the navigation file (optional path).
  -s SP3, --sp3 SP3     Path to the SP3 file (optional path).
  -a ANTEX, --antex ANTEX
                        Path to the ANTEX file (optional path).
  -o OUTPUT, --output OUTPUT
                        Path to the output file (optional path). Use '-' for standard output.
  -t TLIM, --time-limit TLIM
                        Time limit for processing (optional string).
                        Format: (YYYY-MM-DDTHH:MM:SS,YYYY-MM-DDTHH:MM:SS)
  -v, --verbose         Enable verbose output (boolean flag).
  --disable-galileo     Disable Galileo constellation (boolean flag).
  --disable-gps         Disable GPS constellation (boolean flag).
  --disable-signal-travel-time-correction
                        Disable signal travel time correction (boolean flag).
  --disable-earth-rotation-correction
                        Disable Earth rotation correction (boolean flag).
  --disable-tropospheric-correction
                        Disable tropospheric correction (boolean flag).
  --disable-elevation-weighting
                        Disable elevation-based weighting (boolean flag).
  --disable-snr-weighting
                        Disable SNR-based weighting (boolean flag).

Example: geonss --verbose --id WTZR00DEU --navigation-file tests/data/BRDC00IGS_R_20250980000_01D_MN.rnx \
                --measurement-file tests/data/WTZR00DEU_R_20250980000_01D_30S_MO.crx --output-file
```

### Example Library Usage

```python
from geonss import spp
from geonss.parsing import load_cached

# Load the GNSS observations and navigation data.
observation = load_cached('/path/to/observation_file.rnx')
navigation = load_cached('/path/to/navigation_file.rnx')

# Compute the receiver positions with all corrections applied.
result = spp(observation, navigation)

# Print resulting positions and clock bias
print(result)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the GitHub repository.



