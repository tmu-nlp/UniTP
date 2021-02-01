library(ggplot2)

# tri <- read.csv('fscore/triangle.csv')
may <- read.csv('fscore/mary.csv')
tra <- read.csv('fscore/trapezoid.csv')
# fxl <- read.csv('fscore/tri_frozen_xlnet.csv')
txl <- read.csv('fscore/tuned_xlnet.csv')
txm <- read.csv('fscore/mary_tuned_xlnet.csv')

print(head(tra))

# tri$bn <- rep('R/tri', length(tri$wbin))
may$bn <- rep('fastText+BiLSTM', length(may$wbin))
tra$bn <- rep('fastText+BiLSTM', length(tra$wbin))
# fxl$bn <- rep('XLNet\\f', length(fxl$wbin))
txl$bn <- rep('XLNet', length(txl$wbin))
txm$bn <- rep('XLNet', length(txm$wbin))

may$gp <- rep('1', length(may$wbin))
tra$gp <- rep('2', length(tra$wbin))
# fxl$gp <- rep('3', length(fxl$wbin))
txl$gp <- rep('4', length(txl$wbin))
txm$gp <- rep('5', length(txm$wbin))

# tri$type <- rep('/bn', length(tri$wbin))
may$type <- rep('multi-branching', length(may$wbin))
tra$type <- rep('binary', length(tra$wbin))
# fxl$type <- rep('binary', length(fxl$wbin))
txl$type <- rep('binary', length(txl$wbin))
txm$type <- rep('multi-branching', length(txm$wbin))

data <- rbind(may, tra, txl, txm)

p <- ggplot(data, aes(wbin, group = gp, color = gp, linetype = type, shape = bn))
p <- p + guides(color = FALSE, linetype = guide_legend(order = 1), shape = guide_legend(order = 2))
p <- p + geom_line(aes(y = f1), size = 0.7)
p <- p + geom_point(aes(y = f1), size = 2.5)
p <- p + labs(x = 'Sentence Length Bins', y = 'F1 Score')
p <- p + labs(color = "", shape = "")
wlens <- seq(0, 6, 1)
wbins <- paste0(wlens * 10, '-', (wlens + 1) * 10 - 1)
print(wbins)
wbins[length(wbins)] <- paste(wbins[length(wbins)], "  ")
p <- p + scale_x_continuous(breaks = wlens,
                            labels = wbins,
                            expand = c(0.03, 0.03),
                            sec.axis = sec_axis(~.,
                                                breaks = wlens, labels = tra$num, 
                                                name = "Number of Test Samples"))
# p <- p + scale_color_discrete(labels = c('XLNet', 'Trapezoid', 'Triangle'), breaks = c('XLNet', 'Trapezoid', 'Triangle'))
# p <- p + scale_shape_discrete(labels = c('XLNet', 'Trapezoid', 'Triangle'), breaks = c('XLNet', 'Trapezoid', 'Triangle'))
p <- p + theme(legend.position = c(0.29, 0.17),
               legend.direction = "horizontal",
               legend.background = element_blank(),
               legend.key = element_rect(fill = "white", color = NA),
               legend.title = element_blank(),
               legend.margin = margin(t = -10, r = 0, b = 0, l = 0, unit = "pt"),
               # legend.spacing.y = unit(0.1, 'cm'),
            #    legend.title = element_text(size = 10),
               text = element_text(size = 15))
p

ggsave('score.pdf', height = 2.3, width = 5.2, dpi = 600)