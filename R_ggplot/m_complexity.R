library(ggplot2)
library(reshape2)
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

data <- read.csv('m_sent_en.csv')
data$combined <- data$linear + data$square
lin_lm <- lm(linear ~ len + I(len^2), data = data)
add_lm <- lm(combined ~ len + I(len^2), data = data)
max_lm <- lm(max_square ~ len + I(len^2), data = data)
factor_levels <- c("linear", "square", "combined", "max_square")
data <- melt(data = data, id.vars = c("len"), measure.vars = factor_levels)
data$variable <- factor(data$variable, levels = factor_levels, labels = factor_levels)

p <- ggplot(data, aes(len, value))#, alpha = I(0.2))) log-log would not tell lin-sqrt diff
p <- p + scale_fill_continuous(low = "blue", high = "tomato")
# p <- p + scale_fill_gradient2(low = "blue", mid = "green", high = "tomato")
p <- p + coord_cartesian(xlim = c(0, 150), ylim = c(0, 500)) # 10000: linear vs. square
p <- p + labs(x = 'Sentence Length', y = 'Number of Nodes')
# p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
# p <- p + labs(color = "", shape = "")
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 14),
               axis.title.x = element_blank(),
               text = element_text(size = 15))
p <- p + geom_bin2d(binwidth = c(1, 3), aes(fill = sqrt(..count..)))
p <- p + stat_smooth(method = 'lm', formula = y ~ splines::bs(x, 4), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)
p <- p + facet_grid(.~variable)
# p <- p + geom_bin2d(aes(len, square), bins = c(100, 250))

lin_f <- factor('linear', levels = factor_levels, labels = factor_levels)
print(lin_lm)
print(add_lm)
print(max_lm)
p <- p + ann_lm(lin_f, lin_lm)

p

ggsave('m_compress_en.pdf', height = 5, width = 20)