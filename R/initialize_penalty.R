initialize_penalty <- function(penalty_main,
                               penalty_external,
                               nr_x,
                               nc_x,
                               nc_unpen,
                               nr_ext,
                               nc_ext,
                               intercept) {

   names(penalty_external) <- c(
        "penalty_type_ext",
        "quantile_ext",
        "gamma_ext",
        "num_penalty_ext",
        "penalty_ratio_ext",
        "user_penalty_ext",
        "custom_multiplier_ext"
    )

    penalty_obj <- c(penalty_main, penalty_external)
    penalty_obj$penalty_type <- rep(penalty_obj$penalty_type, nc_x)

    # ESK: Add quantiles
    #penalty_obj$quantile <- rep(penalty_obj$quantile, nc_x)

    if (is.null(penalty_obj$penalty_ratio)) {
        if (penalty_obj$user_penalty[1] == 0) {
            if (nr_x > nc_x) {
                penalty_obj$penalty_ratio <- 1e-04
            } else {
                penalty_obj$penalty_ratio <- 0.01
            }
            if (penalty_obj$num_penalty < 3) {
                penalty_obj$num_penalty <- 3
                stop("num_penalty must be at least 3
                     when automatically computing penalty path")
            }
        } else {
            penalty_obj$user_penalty <- rev(sort(penalty_obj$user_penalty))
            penalty_obj$penalty_ratio <- 0.0
        }
    }

    if (is.null(penalty_obj$custom_multiplier)) {
        penalty_obj$custom_multiplier <- rep(1.0, nc_x)
    } else if (length(penalty_obj$custom_multiplier) != nc_x) {
        stop(
            "Length of custom_multiplier (",
            length(penalty_obj$custom_multiplier),
            ") not equal to number of columns in x (",
            nc_x, ")"
        )
    }

    # check penalty object for external
    if (nc_ext > 0) {
        penalty_obj$penalty_type_ext <- rep(penalty_obj$penalty_type_ext, nc_ext)

        # ESK: Add quantiles
        #penalty_obj$quantile_ext <- rep(penalty_obj$quantile_ext, nc_ext)

        if (is.null(penalty_obj$penalty_ratio_ext)) {
            if (penalty_obj$user_penalty_ext[1] == 0) {
                if (nr_ext > nc_ext) {
                    penalty_obj$penalty_ratio_ext <- 1e-04
                } else {
                    penalty_obj$penalty_ratio_ext <- 0.01
                }
                if (penalty_obj$num_penalty_ext < 3) {
                    penalty_obj$num_penalty_ext <- 3
                    stop("num_penalty_ext must be at least
                         3 when automatically computing penalty path")
                }
            } else {
                penalty_obj$user_penalty_ext  <- rev(sort(penalty_obj$user_penalty_ext))
                penalty_obj$penalty_ratio_ext <- 0.0
            }
        }

        if (is.null(penalty_obj$custom_multiplier_ext)) {
            penalty_obj$custom_multiplier_ext <- rep(1.0, nc_ext)
        } else if (length(penalty_obj$custom_multiplier_ext) != nc_ext && nc_ext > 0) {
            stop(
                "Length of custom_multiplier_ext (",
                length(penalty_obj$custom_multiplier_ext),
                ") not equal to number of columns in external (",
                nc_ext, ")"
            )
        }
    } else {
        penalty_obj$penalty_type_ext      <- NULL
        penalty_obj$quantile_ext          <- 0
        penalty_obj$num_penalty_ext       <- 1
        penalty_obj$penalty_ratio_ext     <- 0
        penalty_obj$custom_multiplier_ext <- numeric(0)
    }

    # vectors holding penalty type and multipliers across all variables
    if (intercept[2]) {
        penalty_obj$ptype <- c(
            penalty_obj$penalty_type,
            rep(0.0, nc_unpen),
            0.0,
            penalty_obj$penalty_type_ext
        )
        penalty_obj$cmult <- c(
            penalty_obj$custom_multiplier,
            rep(0.0, nc_unpen),
            0.0,
            penalty_obj$custom_multiplier_ext
        )
    } else {
        penalty_obj$ptype <- c(
            penalty_obj$penalty_type,
            rep(0.0, nc_unpen),
            penalty_obj$penalty_type_ext
        )
        penalty_obj$cmult <- c(
            penalty_obj$custom_multiplier,
            rep(0.0, nc_unpen),
            penalty_obj$custom_multiplier_ext
        )
    }

    return(penalty_obj)
}
