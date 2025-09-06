{
  description = "A GNSS implementation in Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject = {
      url = "path:./pyproject.toml"; # Track pyproject.toml as an input
      flake = false;
    };
  };

  outputs = { self, nixpkgs, pyproject }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    in
    {
      devShells = nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python3 = pkgs.python3;

          # Required libraries for compiled Python packages (like NumPy)
          pythonLDLibPath = pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc
            glibc
            zlib
            openblas
          ]);
        in
        {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              python3
              python3Packages.virtualenv

              # Development
              python3Packages.jupyterlab
              python3Packages.ipython

              # Added for binary compatibility
              autoPatchelfHook
            ];

            shellHook = ''
                export SHELL=${pkgs.bashInteractive}/bin/bash
                export VENV_DIR=".venv"
                export PIP_DISABLE_PIP_VERSION_CHECK=1

                export LD_LIBRARY_PATH="${pythonLDLibPath}"

                if [ ! -d "$VENV_DIR" ]; then
                  echo "Creating virtual environment in $VENV_DIR..."
                  ${python3}/bin/python -m venv $VENV_DIR
                fi

                # Activate the virtual environment
                source "$VENV_DIR/bin/activate"

                echo "Installing package and dependencies from pyproject.toml..."
                pip install --disable-pip-version-check --upgrade -e .

                autoPatchelf
            '';
          };
        }
      );
    };
}
