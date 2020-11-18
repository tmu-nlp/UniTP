library(ggplot2)
library(scales)

data <- read.csv('parse_tiger.csv')
length <- length(data$len)
left <- data[c('len', 'left')]
left$type <- rep('left', length)
right <- data[c('len', 'right')]
right$type <- rep('right', length)
# midin <- data[c('len', 'midin')]
# midin$type <- rep('midin', length)
# midin25 <- data[c('len', 'midin25')]
# midin25$type <- rep('midin25', length)
# midin75 <- data[c('len', 'midin75')]
# midin75$type <- rep('midin75', length)
head <- data[c('len', 'head')]
head$type <- rep('head', length)

names(left) <- c('len', 'size', 'type')
names(right) <- c('len', 'size', 'type')
# names(midin) <- c('len', 'size', 'type')
# names(midin25) <- c('len', 'size', 'type')
# names(midin75) <- c('len', 'size', 'type')
names(head) <- c('len', 'size', 'type')

# data <- rbind(left, midin25, midin, midin75, right)
data <- rbind(left, head, right)
data$type <- factor(data$type, levels = c('left', 'midin25', 'midin', 'head', 'midin75', 'right'), labels = paste(expression(rho), sep = " = ", c(0, 0.25, 0.5, 'head', 0.75, 1)))

p <- ggplot(data, aes(len, size, fill = type))#, alpha = I(0.2)))
# p <- p + coord_cartesian(xlim = c(0, 95), ylim = c(0, 650))
p <- p + scale_x_log10()
p <- p + scale_y_log10(
        breaks = c(10, 100, 1000),
        labels = trans_format("log10", math_format(10^.x)))
p <- p + facet_wrap(~type, ncol = 3)
p <- p + labs(x = 'Sentence Length', y = 'Number of Nodes  ')
# p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
# p <- p + labs(color = "", shape = "")
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 14),
               text = element_text(size = 15))
p <- p + geom_bin2d(bins = c(95, 650))
p <- p + stat_smooth(method = 'lm', formula = y ~ splines::bs(x, 4), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)

ggsave('complexity_dccp.pdf', height = 1.8, width = 5.2)