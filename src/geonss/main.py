import pandas as pd
import argparse
import pathlib
from geonss.position import spp
from geonss.parsing import load_cached, load_cached_antex
from geonss.parsing.write import write_sp3_from_xarray
from geonss.constellation import select_constellations

def main():
    parser = argparse.ArgumentParser(
        description="A single point positioning program for GNSS data. Supports Galileo and GPS. Measurements must be dual frequency.",
        epilog="Example: geonss --verbose --id WTZR00DEU --navigation-file tests/data/BRDC00IGS_R_20250980000_01D_MN.rnx --measurement-file " \
               "tests/data/WTZR00DEU_R_20250980000_01D_30S_MO.crx --output-file output.sp3"
    )
    # --- Required Arguments ---
    parser.add_argument(
        "observation",
        type=pathlib.Path,
        help="Path to RINEX observation file. (required path).",
    )

    # --- Optional Arguments ---
    parser.add_argument(
        "-i", "--identifier",
        type=str,
        dest="id",
        default="L01",
        help="Receiver identifier (default: L01) (optional string).",
    )
    parser.add_argument(
        "-n", "--navigation",
        type=pathlib.Path,
        dest="navigation",
        help="Path to the navigation file (optional path).",
    )
    parser.add_argument(
        "-s", "--sp3",
        type=pathlib.Path,
        dest="sp3",
        help="Path to the SP3 file (optional path).",
    )
    parser.add_argument(
        "-a", "--antex",
        type=pathlib.Path,
        dest="antex",
        help="Path to the ANTEX file (optional path).",
    )
    parser.add_argument(
        "-o", "--output",
        type=pathlib.Path,
        dest="output",
        help="Path to the output file (optional path). Use '-' for standard output.",
    )
    parser.add_argument(
        "-t", "--time-limit",
        type=str,
        dest="tlim",
        help="Time limit for processing (optional string). Format: (YYYY-MM-DDTHH:MM:SS,YYYY-MM-DDTHH:MM:SS)",
    )

    # --- Boolean Flags ---
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable verbose output (boolean flag)."
    )
    parser.add_argument(
        "--disable-galileo",
        action="store_true",
        dest="disable_galileo",
        help="Disable Galileo constellation (boolean flag)."
    )
    parser.add_argument(
        "--disable-gps",
        action="store_true",
        dest="disable_gps",
        help="Disable GPS constellation (boolean flag)."
    )
    parser.add_argument(
        "--disable-signal-travel-time-correction",
        action="store_true",
        dest="disable_signal_travel_time_correction",
        help="Disable signal travel time correction (boolean flag)."
    )
    parser.add_argument(
        "--disable-earth-rotation-correction",
        action="store_true",
        dest="disable_earth_rotation_correction",
        help="Disable Earth rotation correction (boolean flag)."
    )
    parser.add_argument(
        "--disable-tropospheric-correction",
        action="store_true",
        dest="disable_tropospheric_correction",
        help="Disable tropospheric correction (boolean flag)."
    )
    parser.add_argument(
        "--disable-elevation-weighting",
        action="store_true",
        dest="disable_elevation_weighting",
        help="Disable elevation-based weighting (boolean flag)."
    )
    parser.add_argument(
        "--disable-snr-weighting",
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

    if not args.navigation and not (args.sp3 and args.antex):
        print("Error: You must provide either a navigation file or an SP3 + ANTEX file.")

    if args.tlim:
        try:
            time_limit = args.tlim.strip("()").split(",")

            if len(time_limit) != 2:
                raise ValueError("Invalid time limit format. Use (YYYY-MM-DDTHH:MM:SS,YYYY-MM-DDTHH:MM:SS)")

            start_str, end_str = time_limit

            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
        except ValueError:
            print("Error: Invalid time limit format. Use (YYYY-MM-DDTHH:MM:SS,YYYY-MM-DDTHH:MM:SS)")
            return

    try:
        # Load the observation data
        if args.tlim:
            observation = load_cached(args.observation, tlim=(start, end), use=["G", "E"])
        else:
            observation = load_cached(args.observation, use=["G", "E"])

        # Load the navigation data
        if args.navigation and args.tlim:
            navigation = load_cached(args.navigation, tlim=(start, end), use=["G", "E"])
        elif args.navigation:
            navigation = load_cached(args.navigation, use=["G", "E"])
        else:
            navigation = None

        # Load the SP3 data
        if args.sp3 and args.tlim:
            sp3_data = load_cached(args.sp3, tlim=(start, end), use=["G", "E"])
        elif args.sp3:
            sp3_data = load_cached(args.sp3, use=["G", "E"])
        else:
            sp3_data = None

        # Load the ANTEX data
        if args.antex:
            antex_data = load_cached_antex(args.antex)
        else:
            antex_data = None

        observation = select_constellations(observation, galileo=(not args.disable_galileo), gps=(not args.disable_gps))
        navigation = select_constellations(navigation, galileo=(not args.disable_galileo), gps=(not args.disable_gps))

        result = spp(
            observation,
            navigation=navigation,
            sp3=sp3_data,
            antex=antex_data,
            enable_signal_travel_time_correction = not args.disable_signal_travel_time_correction,
            enable_earth_rotation_correction= not args.disable_earth_rotation_correction,
            enable_tropospheric_correction = not args.disable_tropospheric_correction,
            enable_elevation_weighting = not args.disable_elevation_weighting,
            enable_snr_weighting = not args.disable_snr_weighting
        )

        # Add the satellite ID as a coordinate to the result
        result = result.assign_coords(satellite_id=args.id)

        write_sp3_from_xarray(
            result,
            args.output,
            satellite_id=args.id
        )

    except KeyboardInterrupt:
        print("Process interrupted by user.")
        return


if __name__ == "__main__":
    main()