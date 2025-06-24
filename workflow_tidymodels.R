library(tidymodels)
library(themis)
library(arrow)
source('sffs.R')

load_data <- function(path) {
  df <- arrow::read_parquet(path)
  df <- df[, 4:33]
  names(df)[29] <- 'label'
  df$label <- dplyr::recode(df$label, `-1` = 0, `0` = 1, `1` = 2)
  df
}

split_data <- function(df, prop = 0.1) {
  set.seed(42)
  initial_split(df, prop = prop, strata = label)
}

select_features_sffs <- function(X, y, max_features = 8) {
  res <- sffs(as.matrix(X), y, max_features)
  cols <- colnames(X)[res$subset]
  message('Melhores atributos (', length(cols), '): ', paste(cols, collapse=', '))
  cols
}

pca_analysis <- function(df) {
  pcs <- prcomp(df, scale. = TRUE)
  var_exp <- pcs$sdev^2 / sum(pcs$sdev^2)
  print(var_exp)
  print(cumsum(var_exp))
}

apply_pca <- function(train_df, test_df, full_df, threshold = 0.95) {
  rec <- recipe(label ~ ., data = train_df) %>%
    step_normalize(all_predictors()) %>%
    step_pca(all_predictors(), threshold = threshold)
  prep_rec <- prep(rec, training = train_df)
  list(
    bake(prep_rec, new_data = train_df),
    bake(prep_rec, new_data = test_df),
    bake(prep_rec, new_data = full_df)
  )
}

make_recipe <- function(train_df) {
  recipe(label ~ ., data = train_df) %>%
    step_smote(label) %>%
    step_normalize(all_predictors())
}

tune_model <- function(wf, grid, folds, metrics) {
  set.seed(42)
  tune_grid(wf, resamples = folds, grid = grid, metrics = metrics)
}

run_models <- function(models, train_df, folds, metrics) {
  results <- list()
  for (nm in names(models)) {
    spec <- models[[nm]]$spec
    grid <- models[[nm]]$grid
    rec <- make_recipe(train_df)
    wf <- workflow() %>% add_model(spec) %>% add_recipe(rec)
    cat('\n>>> Otimizando', nm, '\n')
    res <- tune_model(wf, grid, folds, metrics)
    best <- select_best(res, metric = 'f_meas')
    final_wf <- finalize_workflow(wf, best)
    fit <- fit(final_wf, data = train_df)
    results[[nm]] <- list(fit = fit, metrics = res)
  }
  results
}

evaluate_on_test <- function(trained, test_df, metrics) {
  res <- purrr::map_dfr(names(trained), function(nm) {
    fit <- trained[[nm]]$fit
    preds <- predict(fit, test_df) %>% bind_cols(test_df)
    m <- metrics(preds, truth = label, estimate = .pred_class)
    m$Model <- nm
    m
  })
  res
}

main <- function() {
  path <- 'df_tk.parquet'
  df <- load_data(path)
  cat('Dados:', nrow(df), 'linhas,', ncol(df), 'colunas\n')

  split <- split_data(df, prop = 0.1)
  train_df <- training(split)
  test_df <- testing(split)

  selected <- select_features_sffs(train_df %>% dplyr::select(-label), train_df$label)
  train_df <- train_df[, c(selected, 'label')]
  test_df  <- test_df[, c(selected, 'label')]
  full_df  <- df[, c(selected, 'label')]

  folds <- vfold_cv(train_df, v = 2, strata = label)
  metrics <- metric_set(accuracy, f_meas, recall, precision, kap)

  models <- list(
    svm_linear = list(
      spec = svm_linear(cost = tune()) %>% set_mode('classification'),
      grid = grid_regular(cost(range(-2,1)), levels = 10)
    ),
    svm_rbf = list(
      spec = svm_rbf(cost = tune(), rbf_sigma = tune()) %>% set_mode('classification'),
      grid = grid_latin_hypercube(cost(range(-2,1)), rbf_sigma(range(-4,0)), size = 10)
    ),
    knn = list(
      spec = nearest_neighbor(neighbors = tune()) %>% set_mode('classification'),
      grid = tibble(neighbors = seq(3, 15, 2))
    ),
    rf = list(
      spec = rand_forest(trees = tune(), min_n = tune()) %>% set_mode('classification'),
      grid = grid_latin_hypercube(trees(range(100,500)), min_n(range(2,20)), size = 10)
    ),
    mlp = list(
      spec = mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% set_mode('classification'),
      grid = grid_latin_hypercube(hidden_units(), penalty(), epochs(range(50,200)), size = 10)
    ),
    adaboost = list(
      spec = boost_tree(trees = tune(), learn_rate = tune()) %>%
        set_mode('classification'),
      grid = grid_latin_hypercube(trees(range(50,200)), learn_rate(), size = 10)
    ),
    naive_bayes = list(
      spec = naive_Bayes() %>% set_mode('classification'),
      grid = tibble()
    )
  )

  trained <- run_models(models, train_df, folds, metrics)

  test_metrics <- evaluate_on_test(trained, test_df, metrics)
  print(test_metrics)

  # PCA variant
  cat('\n>>> PCA dos atributos selecionados\n')
  pca_analysis(train_df %>% select(-label))
  pca_dfs <- apply_pca(train_df, test_df, full_df)
  train_p <- pca_dfs[[1]]
  test_p  <- pca_dfs[[2]]
  full_p  <- pca_dfs[[3]]

  trained_pca <- run_models(models, train_p, folds, metrics)
  test_metrics_pca <- evaluate_on_test(trained_pca, test_p, metrics)
  print(test_metrics_pca)
  cat('\n>>> Script concluido!\n')
}

main()
