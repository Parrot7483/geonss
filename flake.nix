{
  description = "A GNSS implementation in Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    requirements = {
      url = "path:./requirements.txt"; # Track requirements.txt as an input
      flake = false;
    };
  };

  outputs = { self, nixpkgs, requirements }:
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
            glibc
            openblas
            stdenv.cc.cc
            zlib
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

                echo "Installing and upgrading dependencies..."
                pip install --upgrade -r ${requirements}
                pip install -e .

                autoPatchelf
            '';
          };
        }
      );
    };
}
