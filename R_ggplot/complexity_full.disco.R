library(ggplot2)
library(scales)
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
                           len = 37.5, size = 580)
    geom_label(data = ann_text, size = 3.5, label = unname(TeX(labels)), fill = fill)
}

data <- read.csv('parse_tiger.csv')
sub_data <- data[which(data$len > 40),]
length <- length(data$len)
left <- data[c('len', 'left')]
left$type <- rep('left', length)
left_lm <- lm(left ~ len + I(len^2), data = sub_data)
right <- data[c('len', 'right')]
right$type <- rep('right', length)
right_lm <- lm(right ~ len + I(len^2), data = sub_data)
midin25 <- data[c('len', 'midin25')]
midin25$type <- rep('midin25', length)
midin25_lm <- lm(midin25 ~ len + I(len^2), data = sub_data)
midin50 <- data[c('len', 'midin50')]
midin50$type <- rep('midin50', length)
midin50_lm <- lm(midin50 ~ len + I(len^2), data = sub_data)
midin75 <- data[c('len', 'midin75')]
midin75$type <- rep('midin75', length)
midin75_lm <- lm(midin75 ~ len + I(len^2), data = sub_data)
head <- data[c('len', 'head')]
head$type <- rep('head', length)
head_lm <- lm(head ~ len + I(len^2), data = sub_data)

print(midin25_lm)
print(midin75_lm)

names(left) <- c('len', 'size', 'type')
names(right) <- c('len', 'size', 'type')
names(midin50) <- c('len', 'size', 'type')
# names(midin25) <- c('len', 'size', 'type')
# names(midin75) <- c('len', 'size', 'type')
names(head) <- c('len', 'size', 'type')

# factor_labels = unname(TeX(c("$\\rho=0$", "$\\rho=0.25$", "$\\rho=0.5$", "$\\rho=head$", "$\\rho=0.75$", "$\\rho=1$")))
# print(factor_labels)
# data <- rbind(left, midin25, midin, midin75, right)
factor_levels = c('left', 'midin25', 'midin50', 'head', 'midin75', 'right')
factor_labels = paste("rho", c(0, 0.25, 0.5, 'head', 0.75, 1), sep = " = ")

data <- rbind(left, midin50, head, right)
data$type <- factor(data$type, levels = factor_levels, labels = factor_labels)

p <- ggplot(data, aes(len, size, fill = type))#, alpha = I(0.2)))
# p <- p + coord_cartesian(xlim = c(0, 95), ylim = c(0, 650))
# p <- p + scale_x_log10()
# p <- p + scale_y_log10(
#         breaks = c(10, 100, 1000),
#         labels = trans_format("log10", math_format(10^.x)))
p <- p + facet_wrap(~type, ncol = 4)
p <- p + labs(x = 'Sentence Length', y = 'Number of Nodes  ')
# p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
# p <- p + labs(color = "", shape = "")
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 14),
               text = element_text(size = 15))
p <- p + coord_cartesian(xlim = c(0, 75), ylim = c(0, 650))
p <- p + geom_bin2d(bins = c(75, 650))
p <- p + stat_smooth(method = 'lm', formula = y ~ splines::bs(x, 4), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)

left_f <- factor('left', levels = factor_levels, labels = factor_labels)
right_f <- factor('right', levels = factor_levels, labels = factor_labels)
head_f <- factor('head', levels = factor_levels, labels = factor_labels)
midin50_f <- factor('midin50', levels = factor_levels, labels = factor_labels)

p <- p + ann_lm(left_f, left_lm)
p <- p + ann_lm(right_f, right_lm)
p <- p + ann_lm(head_f, head_lm)
p <- p + ann_lm(midin50_f, midin50_lm)

ggsave('complexity_tiger40.pdf', height = 1.8, width = 5.2)