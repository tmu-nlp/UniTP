library(ggplot2)

tri <- read.csv('fscore/triangle.csv')
tra <- read.csv('fscore/trapezoid.csv')
xln <- read.csv('fscore/tri_xlnet.csv')

print(head(tri))

tri$type <- rep('Triangle', length(tri$wbin))
tra$type <- rep('Trapezoid', length(tra$wbin))
xln$type <- rep('XLNet', length(xln$wbin))

data <- rbind(tri, tra, xln)

p <- ggplot(data, aes(wbin, group = type, color = type, shape = type))
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
p <- p + theme(legend.position = c(0.33, 0.12),
               legend.direction = "horizontal",
               legend.background = element_blank(),
               legend.key = element_rect(fill = "white", color = NA),
               legend.title = element_blank(),
            #    legend.title = element_text(size = 10),
               text = element_text(size = 15))

ggsave('score.png', height = 2.3, width = 5.2, dpi = 600)