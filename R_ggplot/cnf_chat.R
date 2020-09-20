library(ggplot2)
library(latex2exp)

# library(showtext)
# font_add_google("Montserrat", "Montserrat")
# windows()
# myFont1 <- "Montserrat"

cnf_data = read.csv('cnf_chat.csv', strip.white = T, colClasses = c('factor', 'factor', 'factor', 'numeric', 'numeric', 'character', 'factor'))
# model, data, pre, cnf, f1, tid, eid
# cnf_data <- data[data$exp == 'cnf',]
print(head(cnf_data))
cnf_data$cnf <- 1 - cnf_data$cnf
cnf_data$uni <- factor(cnf_data$pre)
cnf_data$data <- factor(cnf_data$data, c('ptb', 'ctb', 'ktb'), labels = c('PTB - en', 'CTB - zh', 'KTB - jp'), ordered = F)

p <- ggplot(cnf_data, aes(cnf, f1, color = uni, shape = uni))
p <- p + geom_line(aes(group = uni), size = 0.8)
p <- p + geom_point(size = 4.2)
p <- p + facet_grid(data ~ ., scale = "free_y")
midd <- seq(25, 75, 25)
midd <- paste0('L', (100-midd), 'R', midd)
int_breaks <- function(x, n = 3) pretty(x, n)[pretty(x, n) %% 1 == 0]
p <- p + scale_y_continuous(breaks = int_breaks)
p <- p + scale_x_continuous(breaks = seq(0, 1, 0.25),
                            labels = c('Left', midd, 'Right'),
                            expand = c(0.01, 0.01))
p <- p + scale_color_discrete(labels = c('6-layer', '8-layer'), breaks = c(6, 8))
p <- p + scale_shape_discrete(labels = c('6-layer', '8-layer'), breaks = c(6, 8))
p <- p + labs(x = 'CNF factors and their probabilistic interpolations', y = 'F1 score')
p <- p + theme(legend.position = c(0.13, 0.75),
               legend.direction = "vertical",
               legend.spacing.y = unit(0.1, 'cm'),
               # legend.key.size = unit(0.2, "cm"),
            #    legend.background = element_blank(),
            #    legend.key = element_rect(fill = "transparent", color = "transparent"),
               legend.title = element_text(size = 23, family = "Courier"),
               text = element_text(size = 24),
               axis.title = element_blank(),
               plot.tag.position = c(0.02, 0.98),
               plot.tag = element_text(size = 22))
# windowsFonts(Times=windowsFont("Times New Roman"))
leg <- unname(TeX('BiLSTM$_{cxt}$'))
p <- p + labs(color = leg, shape = leg, tag = "F1")

ggsave('cnf_chat.pdf', height = 4.4, width = 8)