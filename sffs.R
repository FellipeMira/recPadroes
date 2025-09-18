compute_info <- function(Z, x, posY) {
  m <- nrow(x)
  mu <- colMeans(x[, Z, drop = FALSE])
  Sb <- matrix(0, length(Z), length(Z))
  Sw <- matrix(0, length(Z), length(Z))
  for (cls in names(posY)) {
    idx <- posY[[cls]]
    x_cls <- x[idx, Z, drop = FALSE]
    mu_j <- colMeans(x_cls)
    sig_j <- stats::cov(x_cls)
    Sb <- Sb + (length(idx) / m) * tcrossprod(mu - mu_j)
    Sw <- Sw + (length(idx) / m) * sig_j
  }
  det(Sb + Sw) / det(Sw)
}

compute_info_gain_add <- function(Z, W, x, posY) {
  m <- nrow(x)
  vecJ <- numeric(length(W))
  for (i in seq_along(W)) {
    S <- sort(unique(c(Z, W[i])))
    mu <- colMeans(x[, S, drop = FALSE])
    Sb <- matrix(0, length(S), length(S))
    Sw <- matrix(0, length(S), length(S))
    for (cls in names(posY)) {
      idx <- posY[[cls]]
      x_cls <- x[idx, S, drop = FALSE]
      mu_j <- colMeans(x_cls)
      sig_j <- stats::cov(x_cls)
      Sb <- Sb + (length(idx) / m) * tcrossprod(mu - mu_j)
      Sw <- Sw + (length(idx) / m) * sig_j
    }
    vecJ[i] <- det(Sb + Sw) / det(Sw)
  }
  vecJ
}

compute_info_gain_remove <- function(Z, x, posY) {
  m <- nrow(x)
  vecJ <- numeric(length(Z))
  for (item in seq_along(Z)) {
    at <- Z[item]
    S <- Z[-item]
    mu <- colMeans(x[, S, drop = FALSE])
    Sb <- matrix(0, length(S), length(S))
    Sw <- matrix(0, length(S), length(S))
    for (cls in names(posY)) {
      idx <- posY[[cls]]
      x_cls <- x[idx, S, drop = FALSE]
      mu_j <- colMeans(x_cls)
      sig_j <- stats::cov(x_cls)
      Sb <- Sb + (length(idx) / m) * tcrossprod(mu - mu_j)
      Sw <- Sw + (length(idx) / m) * sig_j
    }
    vecJ[item] <- det(Sb + Sw) / det(Sw)
  }
  vecJ
}

sffs <- function(X, y, max_features = ncol(X)) {
  n_feats <- ncol(X)
  max_features <- ifelse(is.null(max_features), n_feats, max_features)
  classes <- unique(y)
  posY <- lapply(classes, function(cls) which(y == cls))
  names(posY) <- classes
  Z <- integer(0)
  W <- seq_len(n_feats)
  best_subset <- integer(0)
  best_score <- -Inf

  while (length(W) > 0 && length(Z) < max_features) {
    add_scores <- compute_info_gain_add(Z, W, X, posY)
    idx_best <- which.max(add_scores)
    f_best <- W[idx_best]
    W <- W[-idx_best]
    Z <- c(Z, f_best)
    J_cur <- add_scores[idx_best]

    improved <- TRUE
    while (improved && length(Z) > 2) {
      remove_scores <- compute_info_gain_remove(Z, X, posY)
      idx_rm <- which.max(remove_scores)
      if (remove_scores[idx_rm] > J_cur) {
        f_rm <- Z[idx_rm]
        Z <- Z[-idx_rm]
        W <- c(W, f_rm)
        J_cur <- remove_scores[idx_rm]
      } else {
        improved <- FALSE
      }
    }
    if (J_cur > best_score) {
      best_score <- J_cur
      best_subset <- Z
    }
  }
  list(subset = best_subset, score = best_score)
}
