library(ggplot2)

de <- read.csv('m_sent_de.csv')
en <- read.csv('m_sent_en.csv')
de$lang <- rep('TIGER', length(de$len))
en$lang <- rep('DPTB', length(en$len))
data <- rbind(de, en)

p <- ggplot(data, aes(len, height))
p <- p + scale_fill_continuous(low = "blue", high = "tomato")
# p <- p + scale_fill_gradient2(low = "blue", mid = "green", high = "tomato")
p <- p + coord_cartesian(xlim = c(0, 150))#, ylim = c(0, 500)) # 10000: linear vs. square
p <- p + labs(x = 'Sentence Length', y = 'Tree Height')
# p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
# p <- p + labs(color = "", shape = "")
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 14),
               axis.title.x = element_blank(),
               text = element_text(size = 15))
p <- p + geom_bin2d(binwidth = c(1, 1), aes(fill = sqrt(..count..)))
p <- p + stat_smooth(method = 'lm', formula = y ~ splines::bs(x, 4), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)
p <- p + facet_grid(lang~.)
# p <- p + geom_bin2d(aes(len, square), bins = c(100, 250))
p

ggsave('m_height.pdf', height = 5, width = 10)