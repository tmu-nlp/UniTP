library(ggplot2)
library(latex2exp)


ptb <- read.csv('parse_ptb_multi.csv')
ctb <- read.csv('parse_ctb_multi.csv')
ktb <- read.csv('parse_ktb_multi.csv')
ptb$corp <- rep('ptb', length(ptb$size))
ctb$corp <- rep('ctb', length(ctb$size))
ktb$corp <- rep('ktb', length(ktb$size))
data <- rbind(ptb, ctb, ktb)
data$corp <- factor(data$corp, levels = c('ptb', 'ctb', 'ktb'), labels = c('PTB', 'CTB', 'KTB'))
data <- data[order(-data$count),]

p <- ggplot(data, aes(size, ratio, size = count, color = log(count), weight = count, alpha = I(0.5)))
p <- p + coord_cartesian(xlim = c(2, NA), ylim = c(0, 1))
p <- p + facet_wrap(corp~., ncol = 3, scale = "free_x")
p <- p + labs(x = 'Layer Size (Multi-branching Treebanks)', y = 'Compress Ratio')
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 55),
               axis.text.y = element_text(size = 45),
               panel.grid.minor.y = element_blank(),
               text = element_text(size = 70),
            #    plot.tag.position = c(0.73, 0.7),
               plot.tag = element_text(size = 40))
p <- p + geom_point()
# p <- p + scale_y_continuous(breaks = c(1/2, 3/5, 2/3, 3/4, 4/5, 8/9, 1), labels = c('1/2', '3/5', '2/3', '3/4', '4/5', '8/9', '1'))
p <- p + scale_x_continuous(breaks = c(2, 50, 100, 150, 200, 250), labels = c(2, 50, 100, 150, 200, '250   '))
p <- p + scale_size_continuous(range = c(1, 30))
p <- p + scale_color_continuous(low = "blue", high = "tomato")
# p <- p + stat_smooth(geom = 'line', alpha = 0.5, color = 'black', show.legend = FALSE, size = 2.5, method = 'rlm')#, n = 100)
p <- p + stat_smooth(method = 'glm', formula = y ~ splines::bs(x, 23), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE, size = 2.5, n = 300)
# p <- p + labs(tag = unname(TeX('$\\leftarrow$ Bottom-up combinatory direction')))
# p <- p + annotate("text", x = 2, y = 0.532, label = "Top", size = 14)
# p <- p + annotate('text', x = 180, y = 0.4, label = unname(TeX('$\\leftarrow$ Bottom-up combinatory direction')), size = 14)
p <- p + annotate("text", x = 160, y = 0.1, label = unname(TeX("|$\\leftarrow$ Bottom Layers from Long Sentences $\\rightarrow$|")), size = 15, color = 'gray')
p

ggsave('level_ratio_multi_full.pdf', height = 8.4, width = 60, limitsize = FALSE)