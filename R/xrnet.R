#' @useDynLib xrnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats predict
#' @importFrom bigmemory is.big.matrix
#' @importFrom methods is
NULL

#' Fit hierarchical regularized regression model
#'
#' @description Fits hierarchical regularized regression model that enables the incorporation of external data
#' for predictor variables. Both the predictor variables and external data can be regularized
#' by the most common penalties (lasso, ridge, elastic net).
#' Solutions are computed across a two-dimensional grid of penalties (a separate penalty path is computed
#' for the predictors and external variables). Currently support regularized linear and logistic regression,
#' future extensions to other outcomes (i.e. Cox regression) will be implemented in the next major update.
#'
#' @param x predictor design matrix of dimension \eqn{n x p}, matrix options include:
#' \itemize{
#'    \item matrix
#'    \item big.matrix
#'    \item filebacked.big.matrix
#'    \item sparse matrix (dgCMatrix)
#' }
#' @param y outcome vector of length \eqn{n}
#' @param external (optional) external data design matrix of dimension \eqn{p x q}, matrix options include:
#' \itemize{
#'     \item matrix
#'     \item sparse matrix (dgCMatrix)
#' }
#' @param unpen (optional) unpenalized predictor design matrix, matrix options include:
#' \itemize{
#'     \item matrix
#' }
#' @param family error distribution for outcome variable, options include:
#' \itemize{
#'     \item "gaussian"
#'     \item "binomial"
#' }
#' @param penalty_main specifies regularization object for x. See \code{\link{define_penalty}} for more details.
#' @param penalty_external specifies regularization object for external. See \code{\link{define_penalty}} for more details.
#' @param weights optional vector of observation-specific weights. Default is 1 for all observations.
#' @param standardize indicates whether x and/or external should be standardized. Default is c(TRUE, TRUE).
#' @param intercept indicates whether an intercept term is included for x and/or external.
#' Default is c(TRUE, FALSE).
#' @param control specifies xrnet control object. See \code{\link{xrnet.control}} for more details.
#'
#' @details This function extends the coordinate descent algorithm of the R package \code{glmnet} to allow the
#' type of regularization (i.e. ridge, lasso) to be feature-specific. This extension is used to enable fitting
#' hierarchical regularized regression models, where external information for the predictors can be included in the
#' \code{external=} argument. In addition, elements of the R package \code{biglasso} are utilized to enable
#' the use of standard R matrices, memory-mapped matrices from the \code{bigmemory} package, or sparse matrices from the \code{Matrix} package.
#'
#' @references
#' Jerome Friedman, Trevor Hastie, Robert Tibshirani (2010).
#' Regularization Paths for Generalized Linear Models via Coordinate Descent.
#' Journal of Statistical Software, 33(1), 1-22. URL http://www.jstatsoft.org/v33/i01/.
#'
#' @references
#' Zeng, Y., and Breheny, P. (2017).
#' The biglasso Package: A Memory- and Computation-Efficient Solver for Lasso Model Fitting with Big Data in R.
#' arXiv preprint arXiv:1701.05936. URL https://arxiv.org/abs/1701.05936.
#'
#' @references
#' Michael J. Kane, John Emerson, Stephen Weston (2013).
#' Scalable Strategies for Computing with Massive Data.
#' Journal of Statistical Software, 55(14), 1-19. URL http://www.jstatsoft.org/v55/i14/.
#'
#' @return A list of class \code{xrnet} with components:
#' \item{beta0}{matrix of first-level intercepts indexed by penalty values}
#' \item{betas}{3-dimensional array of first-level penalized coefficients indexed by penalty values}
#' \item{gammas}{3-dimensional array of first-level non-penalized coefficients indexed by penalty values}
#' \item{alpha0}{matrix of second-level intercepts indexed by penalty values}
#' \item{alphas}{3-dimensional array of second-level external data coefficients indexed by penalty values}
#' \item{penalty}{vector of first-level penalty values}
#' \item{penalty_ext}{vector of second-level penalty values}
#' \item{family}{error distribution for outcome variable}
#' \item{num_passes}{total number of passes over the data in the coordinate descent algorithm}
#' \item{status}{error status for xrnet fitting}
#' \itemize{
#'     \item 0 = OK
#'     \item 1 = Error/Warning
#' }
#' \item{error_msg}{description of error}
#'
#' @examples
#' ### hierarchical regularized linear regression ###
#' data(GaussianExample)
#'
#' ## define penalty for predictors and external variables
#' ## default is ridge for predictors and lasso for external
#' ## see define_penalty() function for more details
#'
#' penMain <- define_penalty(0, num_penalty = 20)
#' penExt <- define_penalty(1, num_penalty = 20)
#'
#' ## fit model with defined regularization
#' fit_xrnet <- xrnet(
#'     x = x_linear,
#'     y = y_linear,
#'     external = ext_linear,
#'     family = "gaussian",
#'     penalty_main = penMain,
#'     penalty_external = penExt
#' )

#' @export
xrnet <- function(x,
                  y,
                  external         = NULL,
                  unpen            = NULL,
                  family           = c("gaussian", "binomial"),
                  penalty_main     = define_penalty(),
                  penalty_external = define_penalty(),
                  weights          = NULL,
                  standardize      = c(TRUE, TRUE),
                  intercept        = c(TRUE, FALSE),
                  control          = list())
{

    # function call
    this.call <- match.call()

    # check error distribution for y
    family <- match.arg(family)

    ## Prepare x and y ##

    # check type of x matrix
    if (is(x, "matrix")) {
        if (typeof(x) != "double")
            stop("x must be of type double")
        mattype_x <- 1
    }
    else if (is.big.matrix(x)) {
        if (bigmemory::describe(x)@description$type != "double")
            stop("x must be of type double")
        mattype_x <- 2
    } else if ("dgCMatrix" %in% class(x)) {
        if (typeof(x@x) != "double")
            stop("x must be of type double")
        mattype_x <- 3
    } else {
        stop("x must be a standard R matrix, big.matrix, filebacked.big.matrix, or dgCMatrix")
    }

    # check type of y
    y <- as.double(drop(y))

    # check dimensions of x and y
    nr_x  <- NROW(x)
    nc_x  <- NCOL(x)
    y_len <- NROW(y)

    if (y_len != nr_x) {
        stop(
            paste0(
                "Length of y (", y_len,
                ") not equal to the number of rows of x (", nr_x,")"
             )
        )
    }

    ## Prepare external ##
    is_sparse_ext = FALSE
    if (!is.null(external)) {

        # check if external is a sparse matrix
        if (is(external, "sparseMatrix")) {
            is_sparse_ext = TRUE
        } else {
            # convert to matrix
            if (!("matrix" %in% class(external))) {
                external <- as.matrix(external)
            }
            if (typeof(external) != "double") {
                stop("external must be of type double")
            }
        }

        # check dimensions
        nr_ext <- NROW(external)
        nc_ext <- NCOL(external)

        if (nc_x != nr_ext) {
            stop(
                paste0("Number of columns in x (", nc_x,
                      ") not equal to the number of rows in external (", nr_ext,
                      ")"
                )
            )
        }
    } else {
        external <- matrix(vector("numeric", 0), 0, 0)
        nr_ext   <- as.integer(0)
        nc_ext   <- as.integer(0)
    }

    ## Prepare unpenalized covariates ##
    if (!is.null(unpen)) {

        # check dimensions
        nc_unpen <- NCOL(unpen)

        if (y_len != NROW(unpen)) {
            stop(
                paste0(
                    "Length of y (", y_len, ") ",
                    "not equal to the number of rows of unpen (", NROW(unpen),")"
                ))
        }

        # convert unpen to matrix
        if (!("matrix" %in% class(unpen))) {
            unpen <- as.matrix(unpen)
        }
        if (typeof(unpen) != "double") {
            stop("unpen must be a numeric matrix of type 'double'")
        }
    } else {
        unpen    <- matrix(vector("numeric", 0), 0, 0)
        nc_unpen <- as.integer(0)
    }

    # set weights
    if (is.null(weights)) {
        weights <- as.double(rep(1, nr_x))
    } else if (length(weights) != y_len) {
        stop(
            paste0(
                "Length of weights (", length(weights),") ",
                "not equal to length of y (", y_len,
                ")"))
    } else if (any(weights < 0)) {
        stop("weights can only contain non-negative values")
    } else {
        weights <- as.double(weights)
    }

    # check penalty objects
    penalty <- initialize_penalty(
        penalty_main     = penalty_main,
        penalty_external = penalty_external,
        nr_x             = nr_x,
        nc_x             = nc_x,
        nc_unpen         = nc_unpen,
        nr_ext           = nr_ext,
        nc_ext           = nc_ext,
        intercept        = intercept
    )

    # check control object
    control <- do.call("xrnet.control", control)
    control <- initialize_control(
        control_obj = control,
        nc_x        = nc_x,
        nc_unpen    = nc_unpen,
        nc_ext      = nc_ext,
        intercept   = intercept
    )

    # fit model
    fit <- fitModelRcpp(
        x                = x,
        mattype_x        = mattype_x,
        y                = y,
        ext              = external,
        is_sparse_ext    = is_sparse_ext,
        fixed            = unpen,
        weights_user     = weights,
        intr             = intercept,
        stnd             = standardize,
        penalty_type     = penalty$ptype,
        cmult            = penalty$cmult,
        quantiles        = penalty$quantiles,
        gamma            = c(0, 0, 0), #ESK: Test
        num_penalty      = c(penalty$num_penalty, penalty$num_penalty_ext),
        penalty_ratio    = c(penalty$penalty_ratio, penalty$penalty_ratio_ext),
        penalty_user     = penalty$user_penalty,
        penalty_user_ext = penalty$user_penalty_ext,
        lower_cl         = control$lower_limits,
        upper_cl         = control$upper_limits,
        family           = family,
        thresh           = control$tolerance,
        maxit            = control$max_iterations,
        ne               = control$dfmax,
        nx               = control$pmax
    )

    # check status of model fit
    if (fit$status %in% c(0, 1)) {

        if (fit$status == 0) {
            fit$status <- "0 (OK)"
        }
        else if (fit$status == 1) {
            fit$status <- "1 (Error/Warning)"
            fit$error_msg <- "Max number of iterations reached"
            warning("Max number of iterations reached")
        }

        # Create arrays ordering coefficients by 1st level penalty / 2nd level penalty
        fit$beta0 <- matrix(
            fit$beta0,
            nrow = penalty$num_penalty,
            ncol = penalty$num_penalty_ext,
            byrow = TRUE
        )

        dim(fit$betas) <- c(nc_x, penalty$num_penalty_ext, penalty$num_penalty)
        fit$betas      <- aperm(fit$betas, c(1, 3, 2))

        if (intercept[2]) {
            fit$alpha0 <- matrix(
                fit$alpha0,
                nrow = penalty$num_penalty,
                ncol = penalty$num_penalty_ext, byrow = TRUE
            )
        } else {
            fit$alpha0 <- NULL
        }

        if (nc_ext > 0) {
            dim(fit$alphas) <- c(nc_ext, penalty$num_penalty_ext, penalty$num_penalty)
            fit$alphas      <- aperm(fit$alphas, c(1, 3, 2))
        } else {
            fit$alphas      <- NULL
            fit$penalty_ext <- NULL
        }

        if (nc_unpen > 0) {
            dim(fit$gammas) <- c(nc_unpen, penalty$num_penalty_ext, penalty$num_penalty)
            fit$gammas      <- aperm(fit$gammas, c(1, 3, 2))
        } else {
            fit$gammas <- NULL
        }
    }

    fit$call <- this.call
    class(fit) <- "xrnet"
    return(fit)
}
