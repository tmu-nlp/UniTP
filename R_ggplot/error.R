library(ggplot2)
library(scales)

data <- read.csv('error.csv')
data$aug <- factor(data$aug, levels = c('even', 'head', 'head+left'), labels = c('even (default)', 'head only', 'head+left.bias'))
data$shf <- factor(data$shf, levels = c('N', 'P'), labels = c('w/o shuffle', 'w/ shuffle'))

p <- ggplot(data, aes(epoch, count / 4730, group = group, color = model, linetype = aug, size = shf))
p <- p + scale_y_log10(breaks = c(0.001, 0.01, 0.1, 1),
                       labels = c('0.1%', '1%', '10%', '100%'),
                       sec.axis = sec_axis(~.,
                                           breaks = c(1/4730, 2/4730, 4/4730, 10/4730, 20/4730, 50/4730, 200/4730, 0.1, 1500/4730, 1),
                                           labels = c(1,2,4, 10, 20, 50, 200, 473, 1500, 4730)))
p <- p + geom_line()
p <- p + labs(x = 'Epoch', y = 'Dev Error Rate (Count)')
p <- p + scale_color_discrete(name = '\n\nVariant')
p <- p + scale_size_discrete(range = c(0.4, 0.8), name = 'Shuffle')
p <- p + scale_linetype_manual(values=c('solid', 'dotted', 'longdash'), name = 'Augment')
p <- p + coord_cartesian(ylim = c(1/4730, 1))
p <- p + theme(legend.spacing.y = unit(0.15, 'cm'),
               legend.key.height = unit(0.35, 'cm'),
               legend.margin = margin(t = 0.1, unit="cm"),
               legend.title = element_text(size = 10),
               axis.title = element_text(size = 10),)
p <- p + guides(color = guide_legend(order = 1),
                size = guide_legend(order = 2),
                linetype = guide_legend(order = 3))
p

ggsave('error.pdf', height = 2.43, width = 8)