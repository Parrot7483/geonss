{
  description = "A GNSS implementation in Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    requirements = {
      url = "path:requirements.txt";  # Track requirements.txt as an input
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, requirements }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python3 = pkgs.python3;
        lib = pkgs.lib;

        # Required libraries for compiled Python packages (like NumPy)
        pythonLDLibPath = pkgs.lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc
          glibc
          zlib
          openblas
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            python3Packages.virtualenv
            stdenv.cc.cc
            glibc
          ];

          shellHook = ''
              export SHELL=${pkgs.bashInteractive}/bin/bash
              export VENV_DIR=".venv"
              export PIP_DISABLE_PIP_VERSION_CHECK=1

              export LD_LIBRARY_PATH="${pythonLDLibPath}:$LD_LIBRARY_PATH"

              if [ ! -d "$VENV_DIR" ]; then
                echo "Creating virtual environment in $VENV_DIR..."
                ${python3}/bin/python -m venv $VENV_DIR
              fi

              # Activate the virtual environment
              source "$VENV_DIR/bin/activate"

              echo "Installing and upgrading dependencies..."
              pip install --quiet --disable-pip-version-check --upgrade -r ${requirements}
              pip install --quiet --disable-pip-version-check -e .
          '';
        };
      }
    );
}
