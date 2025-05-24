{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.notebook
      python-pkgs.jupyter
      python-pkgs.pip
      python-pkgs.sympy
      python-pkgs.uncertainties
      python-pkgs.twine
    ]))
  ];

  # Workaround: make VSCode's Python extension read the .venv
 shellHook = ''
    if [ ! -d .venv ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate
  '';
}
