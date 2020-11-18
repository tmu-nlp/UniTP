library(ggplot2)
library(latex2exp)

data <- read.csv('parse_tiger_head.csv')
data <- data[order(-data$count),]

p <- ggplot(data, aes(size, ratio, size = count, color = count, weight = count, alpha = I(0.5)))
p <- p + coord_cartesian(xlim = c(2, NA), ylim = c(0.47, 1))
# p <- p + facet_wrap(~fct, ncol = 1)
p <- p + labs(x = unname(TeX('Layer Size ($\\rho = head:\\,\\, C_{All} = 0.75_{\\pm 0.14}\\, \\, C_{40+} = 0.77_{\\pm 0.05}$)')), y = 'Compress Ratio')
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 55),
               axis.text.y = element_text(size = 45),
               panel.grid.minor.y = element_blank(),
               text = element_text(size = 70),
               plot.tag.position = c(0.73, 0.7),
               plot.tag = element_text(size = 40))
p <- p + geom_point()
p <- p + scale_y_continuous(breaks = c(1/2, 3/5, 2/3, 3/4, 4/5, 8/9, 1), labels = c('1/2', '3/5', '2/3', '3/4', '4/5', '8/9', '1'))
p <- p + scale_x_continuous(breaks = c(2, 50, 100))
p <- p + scale_size_continuous(range = c(1, 30))
p <- p + scale_color_continuous(low = "blue", high = "tomato")
p <- p + stat_smooth(method = 'lm', formula = y ~ splines::bs(x, 25), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE, size = 2.5, n = 300)
# p <- p + labs(tag = unname(TeX('$\\leftarrow$ Bottom-up combinatory direction')))
p <- p + annotate("text", x = 2, y = 0.532, label = "Top", size = 14)
p <- p + annotate('text', x = 70, y = 0.997, label = unname(TeX('$\\leftarrow$ Zero-combine Layers')), size = 15)
p <- p + annotate('text', x = 90, y = 0.55, label = unname(TeX('$\\leftarrow$ Bottom-up Combinatory Direction')), size = 15)
# p <- p + annotate("text", x = 148, y = 0.5, label = unname(TeX("|$\\leftarrow$ Open Bottom Layers from Long Sentences $\\rightarrow$|")), size = 15, color = 'gray')
# p <- p + annotate("text", x = 240, y = 0.71, label = unname(TeX("$\\leftarrow$ layer@0")), size = 11, color = 'blue')
# p <- p + annotate("text", x = 180, y = 0.713, label = unname(TeX("$\\leftarrow$ layer@1")), size = 11, color = 'blue')
p

ggsave('level_ratio.pdf', height = 8, width = 20)