library(ggplot2)
library(scales)

data <- read.csv('compress_de.csv')

p <- ggplot(data, aes(len, gap))
p <- p + geom_bin2d(binwidth = 1)
p <- p + labs(x = 'Sentence Length Bins', y = 'Tree Gap Degree')
p <- p + scale_fill_gradient(trans = 'log2', # low = "dark gray", high = "dark blue", 
    breaks = trans_breaks("log2", function(x) 2^x),
    labels = trans_format("log2", math_format(2^.x)))
# p <- p + guides(fill = FALSE)

ggsave('gap.pdf', height = 2.5, width = 5.2, dpi = 600)