#' Define regularization object for predictor and external data
#'
#' @description Defines regularization for predictors and external data variables in \code{\link{xrnet}} fitting.
#' Use helper functions define_lasso, define_ridge, or define_enet to specify a common penalty on x or external.
#'
#' @param penalty_type type of regularization. Default is 1 (Lasso).
#' Can supply either a scalar value or vector with length equal to the number of variables the matrix.
#' \itemize{
#'    \item 0 = Ridge
#'    \item (0,1) = Elastic-Net
#'    \item 1 = Lasso / Quantile
#' }
#' @param quantile specifies quantile for quantile penalty. Default of 0.5 reduces to lasso (currently not implemented).
#' @param num_penalty number of penalty values to fit in grid. Default is 20.
#' @param penalty_ratio ratio between minimum and maximum penalty for x.
#' Default is 1e-04 if \eqn{n > p} and 0.01 if \eqn{n <= p}.
#' @param user_penalty user-defined vector of penalty values to use in penalty path.
#' @param custom_multiplier variable-specific penalty multipliers to apply to overall penalty.
#' Default is 1 for all variables. 0 is no penalization.
#'
#' @return A list object with regularization settings that are used to define the regularization
#' for predictors or external data in \code{\link{xrnet}} and \code{\link{tune_xrnet}}:
#' \item{penalty_type}{The penalty type, scalar with value in range [0, 1].}
#' \item{quantile}{Quantile for quantile penalty, 0.5 defaults to lasso (not currently implemented).}
#' \item{num_penalty}{The number of penalty values in the penalty path.}
#' \item{penalty_ratio}{The ratio of the minimum penalty value compared to the maximum penalty value.}
#' \item{user_penalty}{User-defined numeric vector of penalty values, NULL if not provided by user.}
#' \item{custom_multiplier}{User-defined feature-specific penalty multipliers, NULL if not provided by user.}
#'
#' @examples
#'
#' # define ridge penalty with penalty grid split into 30 values
#' my_penalty <- define_penalty(penalty_type = 0, num_penalty = 30)
#'
#' # define elastic net (0.5) penalty with user-defined penalty
#' my_custom_penalty <- define_penalty(penalty_type = 0.5, user_penalty = c(100, 50, 10, 1, 0.1))

penalty_mapper <- function(penalty_type = 1,
                           quantile = 0.5,
                           gamma = 0.0) {
    # Checks:
    if (quantile < 0 | quantile > 1) {
        stop("quantile must be between 0 and 1")
    }
    if (is.numeric(penalty_type)) {
        if (length(penalty_type) > 1) {
            stop("penalty_type must be of length 1")
        }
        if (penalty_type < 0 || penalty_type > 1) {
            stop("If penalty_type is numeric, it must be between 0 and 1")
        }
        quantile     = penalty_type
        penalty_type = 1 # Elastic net regularizer
        gamma = gamma
    } else if (is.character(penalty_type)) {
        if(!(penalty_type %in% c("ridge", "lasso", "enet", "q1", "scad", "mcp"))) {
            stop ("If penalty_type is string, it must be one of 'ridge, lasso, enet, q1, scad, mcp'.")
        }
        if (penalty_type == "ridge") {
            quantile     = 0
            penalty_type = 1
        } else if (penalty_type == "lasso") {
            quantile     = 1
            penalty_type = 1 # Elastic net regularizer
        } else if (penalty_type == "enet") {
            quantile     = quantile
            penalty_type = 1 # Elastic net regularizer
        } else if (penalty_type == "q1") {
            quantile     = quantile
            penalty_type = 2 # Q1 regularizer
        } else if (penalty_type == "scad") {
            if (gamma < 2) {
                warning("For penalty_type = 'scad', gamma parameter must be > 2. Set to 3.7")
                gamma <- 3.7
            } else {
                gamma = gamma
            }
            penalty_type = 3 # SCAD regularizer
        } else if (penalty_type == "mcp") {
            if (gamma < 2) {
                warning("For penalty_type = 'mcp', gamma parameter must be > 1. Set to 3")
                gamma <- 3
            } else {
                gamma = gamma
            }
            penalty_type = 4 # MCP regularizer
        }
    }
    return(list(penalty_type = penalty_type,
                quantile = quantile,
                gamma = gamma))
}
