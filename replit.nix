{ pkgs }: {
  deps = [
    pkgs.python311Full
    pkgs.tk
    pkgs.tcl
    pkgs.xorg.libX11
    pkgs.xorg.libXext
    pkgs.xorg.libXrender
    pkgs.xorg.libXtst
    pkgs.xorg.libXi
    pkgs.zlib
    pkgs.libjpeg
    pkgs.libtiff
    pkgs.libpng
  ];
}
