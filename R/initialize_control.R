initialize_control <- function(control_obj,
                               nc_x,
                               nc_unpen,
                               nc_ext,
                               intercept) {

    if (is.null(control_obj$dfmax)) {
        control_obj$dfmax <- as.integer(nc_x + nc_ext + nc_unpen + intercept[1] + intercept[2])
    } else if (control_obj$dfmax <= 0 || as.integer(control_obj$dfmax) != control_obj$dfmax) {
        stop("dfmax can only contain postive integers")
    }

    if (is.null(control_obj$pmax)) {
        control_obj$pmax <- as.integer(min(2 * control_obj$dfmax + 20, nc_x + nc_ext + nc_unpen + intercept[2]))
    } else if (control_obj$pmax <= 0 || as.integer(control_obj$pmax) != control_obj$pmax) {
        stop("pmax can only contain positive integers")
    }

    if (is.null(control_obj$lower_limits)) {
        control_obj$lower_limits <- rep(-Inf, nc_x + nc_ext + nc_unpen + intercept[2])
    } else if (length(control_obj$lower_limits) != nc_x + nc_ext + nc_unpen) {
        stop(
            "Length of lower_limits (",
            length(control_obj$lower_limits),
            ") not equal to sum of number of columns in x, unpen, and external (",
            nc_x + nc_ext + nc_unpen, ")"
        )
    } else if (intercept[2]) {
        control_obj$lower_limits <- c(
            control_obj$lower_limits[1:(nc_x + nc_unpen)],
            -Inf,
            control_obj$lower_limits[(nc_x + nc_unpen + 1):length(control_obj$lower_limits)]
        )
    }

    if (is.null(control_obj$upper_limits)) {
        control_obj$upper_limits <- rep(Inf, nc_x + nc_ext + nc_unpen + intercept[2])
    } else if (length(control_obj$upper_limits) != nc_x + nc_ext + nc_unpen) {
        stop(
            "Length of upper_limits (",
            length(control_obj$upper_limits),
            ") not equal to sum of number of columns in x, unpen, and external (",
            nc_x + nc_ext + nc_unpen, ")"
        )
    } else if (intercept[2]) {
        control_obj$upper_limits <- c(
            control_obj$upper_limits[1:(nc_x + nc_unpen)],
            -Inf,
            control_obj$upper_limits[(nc_x + nc_unpen + 1):length(control_obj$upper_limits)]
        )
    }
    return(control_obj)
}
