{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }@inputs:
    let
      system = "x86_64-linux";
      pkgs = inputs.nixpkgs.legacyPackages.${system};
      myPython = (
        pkgs.python312.withPackages (
          p: with p; [
            matplotlib
            tkinter
            pyqt6
            pyserial
            numpy
          ]
        )
      );
      env =
        script:
        (pkgs.buildFHSEnv {
          name = "python-env";
          targetPkgs =
            pkgs:
            (with pkgs; [
              myPython
              python312Packages.pip
              python312Packages.virtualenv
              python312Full
              # Support binary wheels from PyPI
              pythonManylinuxPackages.manylinux2014Package
              # Enable building from sdists
              cmake
              ninja
              libgcc
              binutils
              coreutils
              expat
              libz
              gcc
              glib
              zlib
              libGL
              fontconfig
              xorg.libX11
              libxkbcommon
              freetype
              dbus
              pre-commit
            ]);
          multiPkgs =
            pkgs: with pkgs; [
              binutils
              coreutils
              expat
              libz
              gcc
              glib
              zlib
              libGL
            ];
          profile = ''
            export LIBRARY_PATH=/usr/lib:/usr/lib64:$LIBRARY_PATH
          '';
          runScript = "${
            pkgs.writeShellScriptBin "runScript" (
              ''
                    set -e
                    test -d env || ${pkgs.python312Full.interpreter} -m venv env
                source env/bin/activate
                set +e
              ''
              + script
            )
          }/bin/runScript";
        }).env;
    in
    {
      devShells.${system}.default = env "zsh";
    };
}
