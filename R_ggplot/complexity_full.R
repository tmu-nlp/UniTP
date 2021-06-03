library(ggplot2)
library(scales)

data <- read.csv('parse_ptb.csv')
ptb_left_lm <- lm(left ~ len + I(len^2), data = data)
ptb_right_lm <- lm(right ~ len + I(len^2), data = data)
ptb_midin_lm <- lm(midin ~ len + I(len^2), data = data)
ptb_midout_lm <- lm(midout ~ len + I(len^2), data = data)
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

data <- rbind(left, right, midin, midout)
data$corp <- rep('PTB', 4 * length)
ptb <- data

data <- read.csv('parse_ctb.csv')
ctb_left_lm <- lm(left ~ len + I(len^2), data = data)
ctb_right_lm <- lm(right ~ len + I(len^2), data = data)
ctb_midin_lm <- lm(midin ~ len + I(len^2), data = data)
ctb_midout_lm <- lm(midout ~ len + I(len^2), data = data)
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

data <- rbind(left, right, midin, midout)
data$corp <- rep('CTB', 4 * length)
ctb <- data

data <- read.csv('parse_ktb.csv')
ktb_left_lm <- lm(left ~ len + I(len^2), data = data)
ktb_right_lm <- lm(right ~ len + I(len^2), data = data)
ktb_midin_lm <- lm(midin ~ len + I(len^2), data = data)
ktb_midout_lm <- lm(midout ~ len + I(len^2), data = data)
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

data <- rbind(left, right, midin, midout)
data$corp <- rep('KTB', 4 * length)
ktb <- data

multi <- read.csv('parse_multi.csv')
multi_ptb <- multi[which(multi['corp'] == 'PTB'),]
ptb_multi_lm <- lm(size ~ len + I(len^2), data = multi_ptb)
multi_ctb <- multi[which(multi['corp'] == 'CTB'),]
ctb_multi_lm <- lm(size ~ len + I(len^2), data = multi_ctb)
multi_ktb <- multi[which(multi['corp'] == 'KTB'),]
ktb_multi_lm <- lm(size ~ len + I(len^2), data = multi_ktb)

data <- rbind(ptb, ctb, ktb, multi)
data$corp <- factor(data$corp, levels = c('PTB', 'CTB', 'KTB'))
data$type <- factor(data$type, levels = c('left', 'right', 'midin', 'midout', 'none'), labels = c('CNF Left', 'CNF Right', 'nCNF Midin', 'nCNF Midout', 'Multi.'))

p <- ggplot(data, aes(len, size, fill = type))#, alpha = I(0.2)))
# p <- p + scale_x_log10()
# p <- p + scale_y_log10(
#         breaks = c(1, 10, 100, 1000),
#         labels = trans_format("log10", math_format(10^.x)))
p <- p + facet_grid(corp~type)
# p <- p + scale_x_continuous(breaks = c(1, 100, 200))
# p <- p + facet_grid(type~corp, scales = 'free', space = 'free')
p <- p + labs(x = 'Sentence Length', y = 'Number of Nodes  ')
# p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
# p <- p + labs(color = "", shape = "")
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 14),
               text = element_text(size = 15))
p <- p + geom_bin2d(bins = c(150, 650))
p <- p + stat_smooth(method = 'lm', formula = y ~ x + I(x^2), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)

ptb_f <- factor('PTB', levels = c('PTB', 'CTB', 'KTB'))
ctb_f <- factor('CTB', levels = c('PTB', 'CTB', 'KTB'))
ktb_f <- factor('KTB', levels = c('PTB', 'CTB', 'KTB'))
left_f <- factor('left', levels = c('left', 'right', 'midin', 'midout', 'none'), labels = c('CNF Left', 'CNF Right', 'nCNF Midin', 'nCNF Midout', 'Multi.'))
right_f <- factor('right', levels = c('left', 'right', 'midin', 'midout', 'none'), labels = c('CNF Left', 'CNF Right', 'nCNF Midin', 'nCNF Midout', 'Multi.'))
midin_f <- factor('midin', levels = c('left', 'right', 'midin', 'midout', 'none'), labels = c('CNF Left', 'CNF Right', 'nCNF Midin', 'nCNF Midout', 'Multi.'))
midout_f <- factor('midout', levels = c('left', 'right', 'midin', 'midout', 'none'), labels = c('CNF Left', 'CNF Right', 'nCNF Midin', 'nCNF Midout', 'Multi.'))
multi_f <- factor('none', levels = c('left', 'right', 'midin', 'midout', 'none'), labels = c('CNF Left', 'CNF Right', 'nCNF Midin', 'nCNF Midout', 'Multi.'))

# min_z <- min(ptb_left_lm$coefficients[3],
#              ptb_right_lm$coefficients[3],
#              ptb_midin_lm$coefficients[3],
#              ptb_midout_lm$coefficients[3],
#              ptb_multi_lm$coefficients[3],
#              ctb_left_lm$coefficients[3],
#              ctb_right_lm$coefficients[3],
#              ctb_midin_lm$coefficients[3],
#              ctb_midout_lm$coefficients[3],
#              ctb_multi_lm$coefficients[3],
#              ktb_left_lm$coefficients[3],
#              ktb_right_lm$coefficients[3],
#              ktb_midin_lm$coefficients[3],
#              ktb_midout_lm$coefficients[3],
#              ktb_multi_lm$coefficients[3])

# max_z <- max(ptb_left_lm$coefficients[3],
#              ptb_right_lm$coefficients[3],
#              ptb_midin_lm$coefficients[3],
#              ptb_midout_lm$coefficients[3],
#              ptb_multi_lm$coefficients[3],
#              ctb_left_lm$coefficients[3],
#              ctb_right_lm$coefficients[3],
#              ctb_midin_lm$coefficients[3],
#              ctb_midout_lm$coefficients[3],
#              ctb_multi_lm$coefficients[3],
#              ktb_left_lm$coefficients[3],
#              ktb_right_lm$coefficients[3],
#              ktb_midin_lm$coefficients[3],
#              ktb_midout_lm$coefficients[3],
#              ktb_multi_lm$coefficients[3])

# abs_z <- max(max_z, - min_z)
# max_z <- abs_z
# min_z <- - abs_z

library(latex2exp)

ann_lm <- function(corp_f, factor_f, lm_f) {
    lm_2 <- lm_f$coefficients[2]
    lm_3 <- lm_f$coefficients[3]
    lm_2 <- format(round(lm_2, 1), scientific = F, nsmall = 1)
    sign <- if (lm_3 > 0) '+' else '-'
    abs3 <- abs(lm_3)
    abs3 <- format(round(abs3, 4), scientific = F)
    labels <- paste('$', lm_2, 'x', ' ', sign, ' ', abs3, 'x^2$')
    fill <- if (lm_3 > 0) 'coral1' else 'cadetblue1'
    ann_text <- data.frame(corp = corp_f,
                           type = factor_f,
                           len = 130, size = 1875)
    geom_label(data = ann_text, size = 3.5, label = unname(TeX(labels)), fill = fill)
}

p <- p + ann_lm(ptb_f, left_f,   ptb_left_lm)
p <- p + ann_lm(ptb_f, right_f,  ptb_right_lm)
p <- p + ann_lm(ptb_f, midin_f,  ptb_midin_lm)
p <- p + ann_lm(ptb_f, midout_f, ptb_midout_lm)
p <- p + ann_lm(ptb_f, multi_f,  ptb_multi_lm)

p <- p + ann_lm(ctb_f, left_f,   ctb_left_lm)
p <- p + ann_lm(ctb_f, right_f,  ctb_right_lm)
p <- p + ann_lm(ctb_f, midin_f,  ctb_midin_lm)
p <- p + ann_lm(ctb_f, midout_f, ctb_midout_lm)
p <- p + ann_lm(ctb_f, multi_f,  ctb_multi_lm)

p <- p + ann_lm(ktb_f, left_f,   ktb_left_lm)
p <- p + ann_lm(ktb_f, right_f,  ktb_right_lm)
p <- p + ann_lm(ktb_f, midin_f,  ktb_midin_lm)
p <- p + ann_lm(ktb_f, midout_f, ktb_midout_lm)
p <- p + ann_lm(ktb_f, multi_f,  ktb_multi_lm)

# p <- p + ylim(0, 2000)
p <- p + scale_y_continuous(breaks = c(0, 500, 1000, 1500, 2000), labels = c(0, '500', '1K', '1.5K', '2K'), limits = c(0, 2000))

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

ggsave('complexity_full.pdf', height = 4, width = 6.5)