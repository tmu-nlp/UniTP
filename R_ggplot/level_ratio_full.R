library(ggplot2)
library(latex2exp)


left <- read.csv('parse_ptb_left.csv')
right <- read.csv('parse_ptb_right.csv')
left$fct <- rep('left', length(left$size))
right$fct <- rep('right', length(right$size))
# data <- left
# data <- rbind(left, right)
midin <- read.csv('parse_ptb_midin.csv')
midout <- read.csv('parse_ptb_midout.csv')
midin$fct <- rep('midin', length(midin$size))
midout$fct <- rep('midout', length(midout$size))
data <- rbind(left, right, midin, midout)
data$fct <- factor(data$fct, levels = c('left', 'right', 'midin', 'midout'), labels = c('CNF Left', 'CNF Right', 'Non-CNF Midin', 'Non-CNF Midout'))
data <- data[order(-data$count),]

p <- ggplot(data, aes(size, ratio, size = count, color = log(count), weight = count, alpha = I(0.5)))
p <- p + coord_cartesian(xlim = c(2, NA), ylim = c(0.47, 0.98))
p <- p + facet_wrap(~fct, ncol = 1)
p <- p + labs(x = 'Layer Size (Binarized PTB)     ', y = 'Compress Ratio')
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 55),
               axis.text.y = element_text(size = 45),
               panel.grid.minor.y = element_blank(),
               text = element_text(size = 70),
            #    plot.tag.position = c(0.73, 0.7),
               plot.tag = element_text(size = 40))
p <- p + geom_point()
p <- p + scale_y_continuous(breaks = c(1/2, 3/5, 2/3, 3/4, 4/5, 8/9, 1), labels = c('1/2', '3/5', '2/3', '3/4', '4/5', '8/9', '1'))
p <- p + scale_x_continuous(breaks = c(2, 50, 100, 150, 200, 249))
p <- p + scale_size_continuous(range = c(1, 30))
p <- p + scale_color_continuous(low = "blue", high = "tomato")
# p <- p + stat_smooth(geom = 'line', alpha = 0.5, color = 'black', show.legend = FALSE, size = 2.5, method = 'rlm')#, n = 100)
p <- p + stat_smooth(method = 'glm', formula = y ~ splines::bs(x, 23), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE, size = 2.5, n = 300)
# p <- p + labs(tag = unname(TeX('$\\leftarrow$ Bottom-up combinatory direction')))
p <- p + annotate("text", x = 2, y = 0.532, label = "Top", size = 14)
p <- p + annotate('text', x = 180, y = 0.83, label = unname(TeX('$\\leftarrow$ Bottom-up combinatory direction')), size = 14)
p <- p + annotate("text", x = 160, y = 0.5, label = unname(TeX("|$\\leftarrow$ Bottom Layers from Long Sentences $\\rightarrow$|")), size = 15, color = 'gray') # ptb 160/249; ptb 154.5/240; ptb 163/254
# p <- p + annotate("text", x = 240, y = 0.715, label = unname(TeX("$\\leftarrow$ layer@0")), size = 9, color = 'blue')
# p <- p + annotate("text", x = 180, y = 0.718, label = unname(TeX("$\\leftarrow$ layer@1")), size = 9, color = 'blue')
p

ann_lm <- function(corp_f, factor_f) {
    ann_text <- data.frame(type = factor_f,
                           len = 78, size = 70)
    geom_label(data = ann_text, size = 3.5, label = unname(TeX(labels)), fill = fill)
}

ggsave('level_ratio_full_ptb.pdf', height = 28, width = 20)