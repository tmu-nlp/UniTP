library(ggplot2)
library(scales)

data <- read.csv('parse_ptb.csv')
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

data <- rbind(ptb, ctb, ktb, multi)
data$corp <- factor(data$corp, levels = c('PTB', 'CTB', 'KTB'))
data$type <- factor(data$type, levels = c('left', 'right', 'midin', 'midout', 'none'), labels = c('CNF Left', 'CNF Right', 'nCNF Midin', 'nCNF Midout', 'Multi.'))

p <- ggplot(data, aes(len, size, fill = type))#, alpha = I(0.2)))
p <- p + scale_x_log10()
p <- p + scale_y_log10(
        breaks = c(1, 10, 100, 1000),
        labels = trans_format("log10", math_format(10^.x)))
p <- p + facet_grid(corp~type)
p <- p + labs(x = 'Sentence Length', y = 'Number of Nodes  ')
# p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
# p <- p + labs(color = "", shape = "")
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 14),
               text = element_text(size = 15))
p <- p + geom_bin2d(bins = c(150, 650))
p <- p + stat_smooth(method = 'lm', formula = y ~ splines::bs(x, 4), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)
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

ggsave('complexity_log.pdf', height = 3, width = 6.5)