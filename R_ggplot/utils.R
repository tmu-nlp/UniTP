library(ggplot2)
library(latex2exp)
folder <- 'stat.data/'

get_orifs <- function(factor_levels, factor_labels) {
    orifs <- list()
    for (fl in factor_levels) {
        orifs[[fl]] <- factor(fl, levels = factor_levels, labels = factor_labels)
    }
    orifs
}

ann_lm <- function(data, factor_f, x, y) {
    lm_f <- lm(size ~ len + I(len^2), subset(data, orif == factor_f))
    lm_2 <- lm_f$coefficients[2]
    lm_3 <- lm_f$coefficients[3]
    lm_2 <- format(round(lm_2, 1), scientific = F, nsmall = 1)
    sign <- if (lm_3 > 0) '+' else '-'
    abs3 <- abs(lm_3)
    abs3 <- format(round(abs3, 4), scientific = F)
    labels <- paste('$', lm_2, 'x', ' ', sign, ' ', abs3, 'x^2$')
    fill <- if (lm_3 > 0) 'coral1' else 'cadetblue1'
    ann_text <- data.frame(orif = factor_f, len = x, size = y)
    geom_label(data = ann_text, size = 3.5, label = unname(TeX(labels)), fill = fill)
}

load_corp <- function(prefix, corp) {
    data <- read.csv(paste0(folder, prefix, corp, '.csv'))
    data$corp <- rep(corp, nrow(data))
    data
}