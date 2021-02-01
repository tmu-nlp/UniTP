library(ggplot2)

data <- read.csv('pca.nccp.csv')
data$label <- factor(data$label, c('NP', 'PP', 'VP', 'S+VP', 'S'), labels = c('NP', 'PP', 'VP', 'S+VP', 'S'), ordered = T)
data <- data[order(data$label),]


p <- ggplot(data, aes(pc2, pc1, color = label))
p <- p + geom_point()
p <- p + theme(legend.position = "none", axis.title = element_blank())
p

ggsave('pca.png', height = 6, width = 6, dpi = 300)