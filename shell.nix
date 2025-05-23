{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.notebook
      python-pkgs.jupyter
      python-pkgs.pip
      python-pkgs.sympy
 

    ]))
  ];

  # Workaround: make VSCode's Python extension read the .venv
  shellHook = ''
    venv="$(cd $(dirname $(which python)); cd ..; pwd)"
    ln -Tsf "$venv" .venv
  '';
}
