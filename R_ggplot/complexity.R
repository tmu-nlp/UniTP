library(ggplot2)

data <- read.csv('parse_ptb.csv')
multi <- read.csv('parse_multi.csv')
left_lm <- lm(left ~ len + I(len^2), data = data)
right_lm <- lm(right ~ len + I(len^2), data = data)
midin_lm <- lm(midin ~ len + I(len^2), data = data)
midout_lm <- lm(midout ~ len + I(len^2), data = data)
print(left_lm$coefficients)
print(right_lm$coefficients)
print(midin_lm$coefficients)
print(midout_lm$coefficients)

length <- length(data$len)
left <- data[c('len', 'left')]
left$type <- rep('left', length)
right <- data[c('len', 'right')]
right$type <- rep('right', length)
midin <- data[c('len', 'midin')]
midin$type <- rep('midin', length)
midout <- data[c('len', 'midout')]
midout$type <- rep('midout', length)

names(left) <- c('len', 'size', 'type')
names(right) <- c('len', 'size', 'type')
names(midin) <- c('len', 'size', 'type')
names(midout) <- c('len', 'size', 'type')

multi <- multi[which(multi$corp == 'PTB'),]
multi <- multi[c('len', 'size', 'type')]
multi_lm <- lm(size ~ len + I(len^2), data = multi)
data <- rbind(left, right, midin, midout, multi)
factor_levels <- c('left', 'right', 'none', 'midin', 'midout')
factor_labels <- c('CNF Left', 'CNF Right', 'Multi-branching', 'non-CNF Midin', 'non-CNF Midout')
data$type <- factor(data$type, levels = factor_levels, labels = factor_labels)

p <- ggplot(data, aes(len, size, fill = type))#, alpha = I(0.2)))
p <- p + coord_cartesian(xlim = c(0, 100), ylim = c(0, 650))
p <- p + facet_wrap(~type, ncol = 3)
p <- p + labs(x = 'Sentence Length', y = 'Number of Nodes')
# p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
# p <- p + labs(color = "", shape = "")
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 14),
               text = element_text(size = 15))
p <- p + geom_bin2d(bins = c(100, 650))
p <- p + stat_smooth(method = 'lm', formula = y ~ x + I(x^2), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)


library(latex2exp)

ann_lm <- function(factor_f, lm_f) {
    lm_2 <- lm_f$coefficients[2]
    lm_3 <- lm_f$coefficients[3]
    lm_2 <- format(round(lm_2, 1), scientific = F, nsmall = 1)
    sign <- if (lm_3 > 0) '+' else '-'
    abs3 <- abs(lm_3)
    abs3 <- format(round(abs3, 4), scientific = F)
    labels <- paste('$', lm_2, 'x', ' ', sign, ' ', abs3, 'x^2$')
    fill <- if (lm_3 > 0) 'coral1' else 'cadetblue1'
    ann_text <- data.frame(type = factor_f,
                           len = 40, size = 550)
    geom_label(data = ann_text, size = 3.5, label = unname(TeX(labels)), fill = fill)
}

left_f <- factor('left', levels = factor_levels, labels = factor_labels)
right_f <- factor('right', levels = factor_levels, labels = factor_labels)
midin_f <- factor('midin', levels = factor_levels, labels = factor_labels)
midout_f <- factor('midout', levels = factor_levels, labels = factor_labels)
multi_f <- factor('none', levels = factor_levels, labels = factor_labels)

p <- p + ann_lm(left_f,   left_lm)
p <- p + ann_lm(right_f,  right_lm)
p <- p + ann_lm(midin_f,  midin_lm)
p <- p + ann_lm(midout_f, midout_lm)
p <- p + ann_lm(multi_f,  multi_lm)

p

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

ggsave('complexity.pdf', height = 3.2, width = 5.2)