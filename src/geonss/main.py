import argparse
import pathlib

def main():
    parser = argparse.ArgumentParser(
        description="A single point positioning program for GNSS data. Supports Galileo and GPS. Measurements must be dual frequency.",
        epilog="Example: geonss --verbose --id WTZR00DEU --navigation-file tests/data/BRDC00IGS_R_20250980000_01D_MN.rnx --measurement-file " \
               "tests/data/WTZR00DEU_R_20250980000_01D_30S_MO.crx --output-file output.sp3"
    )
    # --- Required Arguments ---
    parser.add_argument(
        "observation-file",
        type=pathlib.Path,
        help="Path to RINEX observation file. (required path).",
    )

    # --- Optional Arguments ---
    parser.add_argument(
        "-i", "--identifier",
        type=str,
        dest="id",
        default="sat",
        help="Receiver identifier (default: sat) (optional string).",
    )
    parser.add_argument(
        "-n", "--navigation-file",
        type=pathlib.Path,
        dest="navigation_file",
        help="Path to the navigation file (optional path).",
    )
    parser.add_argument(
        "-s", "--sp3-file",
        type=pathlib.Path,
        dest="sp3_file",
        help="Path to the SP3 file (optional path).",
    )
    parser.add_argument(
        "-a", "--antex-file",
        type=pathlib.Path,
        dest="antex_file",
        help="Path to the ANTEX file (optional path).",
    )
    parser.add_argument(
        "-o", "--output-file",
        type=pathlib.Path,
        dest="output_file_path",
        help="Path to the output file (optional path).",
    )

    # --- Boolean Flags ---
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable verbose output (boolean flag)."
    )
    parser.add_argument(
        "-t", "--disable-signal-travel-time-correction",
        action="store_true",
        dest="disable_signal_travel_time_correction",
        help="Disable signal travel time correction (boolean flag)."
    )
    parser.add_argument(
        "-e", "--disable-earth-rotation-correction",
        action="store_true",
        dest="disable_earth_rotation_correction",
        help="Disable Earth rotation correction (boolean flag)."
    )
    parser.add_argument(
        "-p", "--disable-tropospheric-correction",
        action="store_true",
        dest="disable_tropospheric_correction",
        help="Disable tropospheric correction (boolean flag)."
    )
    parser.add_argument(
        "-w", "--disable-elevation-weighting",
        action="store_true",
        dest="disable_elevation_weighting",
        help="Disable elevation-based weighting (boolean flag)."
    )
    parser.add_argument(
        "-r", "--disable-snr-weighting",
        action="store_true",
        dest="disable_snr_weighting",
        help="Disable SNR-based weighting (boolean flag)."
    )


    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
        parser.print_help()
        return

    if not args.sp3_file and not args.navigation_file:
        print("Error: You must provide either a navigation file or an SP3 file.")

    if args.sp3_file and not args.antex_file:
        print("Warning: You provided an SP3 orbit file but no ANTEX file. Phase center offsets will not be applied.")

    if not args.output_file_path:
        print("Warning: No output file path provided. Defaulting to STDOUT.")

    if args.verbose:
        print("Verbose mode is ON. More details will be printed.")

    print("Program finished.")

if __name__ == "__main__":
    main()