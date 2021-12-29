source('data.R')
# # factor_labels = unname(TeX(c("$\\rho=0$", "$\\rho=0.25$", "$\\rho=0.5$", "$\\rho=head$", "$\\rho=0.75$", "$\\rho=1$")))
# # print(factor_labels)
# # data <- rbind(left, midin25, midin, midin75, right)
# factor_levels = c('left', 'midin25', 'midin50', 'head', 'midin75', 'right')
factor_levels <- seq(25, 75, 25)
factor_levels <- paste0('midin', factor_levels)
factor_levels <- c('left', factor_levels, 'right', 'head', 'multi')
factor_labels <- paste("rho", c(0, 0.25, 0.5, 0.75, 1, 'head'), sep = " = ")
factor_labels <- c(factor_labels, 'Multi-branch')
orifs <- get_orifs(factor_levels, factor_labels)

draw <- function(corp) {
    data <- read.csv(paste0(folder, 'sent.orif-node.', corp, '.csv'))
    data$orif <- factor(data$orif, levels = factor_levels, labels = factor_labels)

    p <- ggplot(data, aes(len, size, fill = orif))#, alpha = I(0.2)))
    p <- p + coord_cartesian(xlim = c(0, 75), ylim = c(0, 650))
    p <- p + facet_wrap(~orif, ncol = 5)
    p <- p + labs(x = 'Sentence Length', y = 'Number of Nodes')
    # p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
    # p <- p + labs(color = "", shape = "")
    p <- p + theme(legend.position = "none",
                   axis.title = element_text(size = 14),
                   text = element_text(size = 15))
    p <- p + geom_bin2d(bins = c(75, 650))
    p <- p + stat_smooth(method = 'lm', formula = y ~ x + I(x^2), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)

    for (orif in orifs) {
        p <- p + ann_lm(data, orif, 40, 550)
    }
    ggsave(paste0(folder, 'complexity.', corp, '.pdf'), height = 3.2, width = 6.4)
}

# p <- p + coord_cartesian(xlim = c(0, 75), ylim = c(0, 650))
# p <- p + geom_bin2d(bins = c(75, 650))

draw('tiger')
draw('dptb')