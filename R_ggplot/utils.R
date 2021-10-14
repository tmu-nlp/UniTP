library(ggplot2)

load_corp <- function(prefix, corp) {
    fname <- paste(prefix, corp, 'csv', sep = '.')
    data <- read.csv(paste0(folder, fname))
    data$corp <- rep(corp, nrow(data))
    data
}