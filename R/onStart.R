.onAttach <- function(libname, pkgname) {
    pkg.version <- read.dcf(file = system.file("DESCRIPTION", package = pkgname),
                            fields = "Version")
    packageStartupMessage(paste("\n",
                                paste(pkgname,
                                      pkg.version,
                                      "is installed!"),
                                "\n",
                                paste0(pkgname, " is built on R ", version$major, ".", version$minor, ".")))
}
