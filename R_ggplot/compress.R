library(ggplot2)
library(scales)

data <- read.csv('compress_en.csv')

p <- ggplot(data, aes(len, rf))
p <- p + geom_bin2d(binwidth = 1)
# p <- p + labs(x = 'Sentence Length Bins', y = 'Tree Height (disco PENN)')
p <- p + labs(x = 'Sentence Length Bins', y = '# Tree Nodes (disco PENN)')
p <- p + scale_fill_gradient(low = "dark gray", high = "dark blue", trans = 'log2',
    breaks = trans_breaks("log2", function(x) 2^x),
    labels = trans_format("log2", math_format(2^.x)))
# p <- p + guides(fill = FALSE)

ggsave('compress.pdf', height = 2.5, width = 5.2, dpi = 600)