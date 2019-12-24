library(ggplot2)

tri_data = read.csv('speed/Triangle.Length.csv')
data = read.csv('speed/Triangle.SPS.csv')
names(tri_data) <- c('unused', 'step', 'length')
names(data) <- c('unused', 'step', 'speed')
tri_data$speed <- data$speed
print(head(tri_data))
tri_data$type <- rep('Triangle', length(tri_data$speed))


trapo_data = read.csv('speed/Trapezoid.Length.csv')
data = read.csv('speed/Trapezoid.SPS.csv')
names(trapo_data) <- c('unused', 'step', 'length')
names(data) <- c('unused', 'step', 'speed')
trapo_data$speed <- data$speed
print(head(trapo_data))
trapo_data$type <- rep('Trapezoid', length(trapo_data$speed))

data <- rbind(tri_data, trapo_data)

p <- ggplot(data, aes(length, speed, color = type, shape = type, alpha = I(0.7)))
p <- p + labs(x = 'Training Batch Length', y = 'Training Speed (sent./sec.)')
p <- p + labs(color = "Data Format", shape = "Data Format")
p <- p + theme(legend.position = c(0.89, 0.84))
            #    legend.key = element_rect(fill = NA, color = NA))
p <- p + geom_point()

# model, data, pre, cnf, f1, tid, eid
# cnf_data <- data[data$exp == 'cnf',]
# print(head(cnf_data))
# cnf_data$cnf <- 1 - cnf_data$cnf
# cnf_data$uni <- factor(cnf_data$pre)
# cnf_data$data <- factor(cnf_data$data, c('ptb', 'ctb', 'ktb'), labels = c('PTB (en)', 'CTB (zh)', 'KTB (jp)'), ordered = F)

# p <- ggplot(cnf_data, aes(cnf, f1, color = uni, shape = uni))
# p <- p + geom_point() + geom_line(aes(group = uni))
# p <- p + facet_grid(data ~ ., scale = "free_y")
# mid <- seq(1, 9, 1)
# mid <- paste0('L', (10-mid), '0R', mid, '0')
# p <- p + scale_x_continuous(breaks = seq(0, 1, 0.1), labels = c('Left', mid, 'Right'))
# p <- p + labs(x = 'CNF factors and their probabilistic interpolations', y = 'F1 score')
# p <- p + theme(legend.position = "none", text = element_text(size = 15), axis.title = element_blank())
# p

ggsave('speed.png', height = 2.7, width = 5.2, dpi = 600)