library(ggplot2)

tri <- read.csv('fscore/triangle.csv')
tra <- read.csv('fscore/trapezoid.csv')
fxl <- read.csv('fscore/tri_frozen_xlnet.csv')
txl <- read.csv('fscore/tuned_xlnet.csv')

print(head(tri))

tri$fmt <- rep('R/tri', length(tri$wbin))
tra$fmt <- rep('R/tra', length(tra$wbin))
fxl$fmt <- rep('X\\f', length(fxl$wbin))
txl$fmt <- rep('X\\t', length(txl$wbin))

tri$type <- rep('/fmt', length(tri$wbin))
tra$type <- rep('/fmt', length(tra$wbin))
fxl$type <- rep('\\tune', length(fxl$wbin))
txl$type <- rep('\\tune', length(txl$wbin))

data <- rbind(tri, tra, fxl, txl)

p <- ggplot(data, aes(wbin, group = fmt, color = fmt, shape = fmt, linetype = type))
p <- p + geom_line(aes(y = f1), size = 0.7)
p <- p + geom_point(aes(y = f1), size = 2.5)
p <- p + labs(x = 'Sentence Length Bins', y = 'F1 Score')
p <- p + labs(color = "", shape = "")
wlens <- seq(0, 6, 1)
wbins <- paste0(wlens * 10, '-', (wlens + 1) * 10 - 1)
print(wbins)
p <- p + scale_x_continuous(breaks = wlens,
                            labels = wbins,
                            expand = c(0.03, 0.03),
                            sec.axis = sec_axis(~.,
                                                breaks = wlens, labels = tri$num, 
                                                name = "Number of Test Samples"))
# p <- p + scale_color_discrete(labels = c('XLNet', 'Trapezoid', 'Triangle'), breaks = c('XLNet', 'Trapezoid', 'Triangle'))
# p <- p + scale_shape_discrete(labels = c('XLNet', 'Trapezoid', 'Triangle'), breaks = c('XLNet', 'Trapezoid', 'Triangle'))
p <- p + theme(legend.position = c(0.31, 0.17),
               legend.direction = "horizontal",
               legend.background = element_blank(),
               legend.key = element_rect(fill = "white", color = NA),
               legend.title = element_blank(),
               legend.margin = margin(t = -10, r = 0, b = 0, l = 0, unit = "pt"),
               # legend.spacing.y = unit(0.1, 'cm'),
            #    legend.title = element_text(size = 10),
               text = element_text(size = 15))

ggsave('score.pdf', height = 2.3, width = 5.2, dpi = 600)