source('data.R')
factor_levels <- c('left', 'right', 'multi', 'midin', 'midout')
factor_labels <- c('CNF Left', 'CNF Right', 'Multi-branching', 'non-CNF Midin', 'non-CNF Midout')
orifs <- get_orifs(factor_levels, factor_labels)

draw <- function(corp) {
    data <- read.csv(paste0(folder, 'sent.orif-node.', corp, '.csv'))
    data$orif <- factor(data$orif, levels = factor_levels, labels = factor_labels)

    p <- ggplot(data, aes(len, size, fill = orif))#, alpha = I(0.2)))
    p <- p + coord_cartesian(xlim = c(0, 100), ylim = c(0, 650))
    p <- p + facet_wrap(~orif, ncol = 3)
    p <- p + labs(x = 'Sentence Length', y = 'Number of Nodes')
    # p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
    # p <- p + labs(color = "", shape = "")
    p <- p + theme(legend.position = "none",
                   axis.title = element_text(size = 14),
                   text = element_text(size = 15))
    p <- p + geom_bin2d(bins = c(100, 650))
    p <- p + stat_smooth(method = 'lm', formula = y ~ x + I(x^2), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)

    for (orif in orifs) {
        p <- p + ann_lm(data, orif, 40, 550)
    }
    ggsave(paste0(folder, 'complexity.', corp, '.pdf'), height = 3.2, width = 5.2)
}

draw('ptb')
draw('ctb')
draw('ktb')
# model, data, pre, cnf, f1, tid, eid
# cnf_data <- data[data$exp == 'cnf',]
# print(head(cnf_data))
# cnf_data$cnf <- 1 - cnf_data$cnf
# cnf_data$uni <- factor(cnf_data$pre)
# cnf_data$data <- factor(cnf_data$data, c('ptb', 'ctb', 'ktb'), labels = c('PTB (en)', 'CTB (zh)', 'KTB (jp)'), ordered = F)

# p <- ggplot(cnf_data, aes(cnf, f1, color = uni, shape = uni))
# p <- p + geom_point() + geom_line(aes(group = uni))
# p <- p + facet_grid(data ~ ., scale = "free_y")
# mid <- seq(1, 9, 1)
# mid <- paste0('L', (10-mid), '0R', mid, '0')
# p <- p + scale_x_continuous(breaks = seq(0, 1, 0.1), labels = c('Left', mid, 'Right'))
# p <- p + labs(x = 'CNF factors and their probabilistic interpolations', y = 'F1 score')
# p <- p + theme(legend.position = "none", text = element_text(size = 15), axis.title = element_blank())
# p
