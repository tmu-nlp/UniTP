library(ggplot2)

cnf_data = read.csv('ablation.csv', strip.white = T, colClasses = c('factor', 'factor', 'factor', 'numeric', 'numeric', 'character', 'factor'))
# model, data, pre, cnf, f1, tid, eid
# cnf_data <- data[data$exp == 'cnf',]
print(head(cnf_data))
cnf_data$cnf <- 1 - cnf_data$cnf
cnf_data$uni <- factor(cnf_data$pre)
cnf_data$data <- factor(cnf_data$data, c('ptb', 'ctb', 'ktb'), labels = c('PTB (en)', 'CTB (zh)', 'KTB (jp)'), ordered = F)

p <- ggplot(cnf_data, aes(cnf, f1, color = uni, shape = uni))
p <- p + geom_point() + geom_line(aes(group = uni))
p <- p + facet_grid(data ~ ., scale = "free_y")
mid <- seq(1, 9, 1)
mid <- paste0('L', (10-mid), '0R', mid, '0')
p <- p + scale_x_continuous(breaks = seq(0, 1, 0.1), labels = c('Left', mid, 'Right'))
p <- p + labs(x = 'CNF factors and their probabilistic interpolations', y = 'F1 score')
p <- p + theme(legend.position = "none", text = element_text(size = 15), axis.title = element_blank())
p

ggsave('cnf_chat.png', height = 4, width = 8, dpi = 300)