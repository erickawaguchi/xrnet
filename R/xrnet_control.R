#' Control function for xrnet fitting
#'
#' @description Control function for \code{\link{xrnet}} fitting.
#'
#' @param tolerance positive convergence criterion. Default is 1e-08.
#' @param max_iterations maximum number of iterations to run coordinate gradient descent
#' across all penalties before returning an error. Default is 1e+05.
#' @param dfmax maximum number of variables allowed in model. Default
#' is \eqn{ncol(x) + ncol(unpen) + ncol(external) + intercept[1] + intercept[2]}.
#' @param pmax maximum number of variables with nonzero coefficient estimate.
#' Default is \eqn{min(2 * dfmax + 20, ncol(x) + ncol(unpen) + ncol(external) + intercept[2])}.
#' @param lower_limits vector of lower limits for each coefficient. Default is -Inf for all variables.
#' @param upper_limits vector of upper limits for each coefficient. Default is Inf for all variables.
#'
#' @return A list object with the following components:
#' \item{tolerance}{The coordinate descent stopping criterion.}
#' \item{dfmax}{The maximum number of variables that will be allowed in the model.}
#' \item{pmax}{The maximum number of variables with nonzero coefficient estimate.}
#' \item{lower_limits}{Feature-specific numeric vector of lower bounds for coefficient estimates}
#' \item{upper_limits}{Feature-specific numeric vector of upper bounds for coefficient estimates}

#' @export
xrnet.control <- function(tolerance = 1e-08,
                          max_iterations = 1e+05,
                          dfmax = NULL,
                          pmax = NULL,
                          lower_limits = NULL,
                          upper_limits = NULL) {

    if (tolerance <= 0) {
        stop("tolerance must be greater than 0")
    }

    if (max_iterations <= 0 || as.integer(max_iterations) != max_iterations) {
        stop("max_iterations must be a positive integer")
    }

    control_obj <- list(
        tolerance = tolerance,
        max_iterations = max_iterations,
        dfmax = dfmax,
        pmax = pmax,
        lower_limits = lower_limits,
        upper_limits = upper_limits
    )
}
