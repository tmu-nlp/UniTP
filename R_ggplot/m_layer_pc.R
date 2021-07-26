library(ggplot2)
library(reshape2)
# len,height,num.comp,comp.size
# data <- read.csv('m_layer_en.csv')
data <- read.csv('m_layer_de.csv')
# de$lang <- rep('TIGER', length(de$len))
# en$lang <- rep('DPTB', length(en$len))
# data <- rbind(de, en)

p <- ggplot(data, aes(comp.size -.001, num.comp -.001)) # TODO: shift to 0
p <- p + scale_fill_continuous(low = "blue", high = "tomato")
# p <- p + scale_fill_gradient2(low = "blue", mid = "green", high = "tomato")
p <- p + labs(x = '# of Children', y = '# of Parents')
# p <- p + labs(x = 'Training Batch Length', y = 'Speed (sents/sec)')
# p <- p + labs(color = "", shape = "")
p <- p + theme(legend.position = "none",
               axis.title = element_text(size = 14),
            #    axis.title.x = element_blank(),
               text = element_text(size = 15))
p <- p + geom_bin2d(binwidth = c(1, 1), aes(fill = ..count..))
# p <- p + stat_smooth(method = 'lm', formula = y ~ splines::bs(x, 4), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)
p

ggsave('m_pc_de.pdf', height = 2.6, width = 5)
# ggsave('m_pc_en.pdf', height = 5, width = 5)