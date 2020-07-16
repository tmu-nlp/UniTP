library(ggplot2)

data = read.csv('f1.losses.csv')
data$x <- data$tag - data$label
data$y <- data$orient
data

p <- ggplot(data, aes(x, y, color = f1, size = f1))
p <- p + labs(x = 'Difference of Tag and Label Loss Coefficients     ', y = 'Orientation Loss Coefficient')
p <- p + geom_point()#size = 15)
# p <- p + geom_hex()#stat = 'identity')
p <- p + scale_x_continuous(limits = c(-0.8, 1.1), breaks = c(-0.7, -0.3, 0, 0.3, 0.7), labels = c('Max.Label', -0.3, 0, 0.3, 'Max.Tag'), expand = c(0, 0))
p <- p + scale_y_continuous(expand = c(0.03, 0.03), breaks = c(0.1, 0.3, 0.5, 0.7))
p <- p + scale_color_gradient(low = "darkblue", high = "tomato", na.value = NA)
p <- p + guides(size = FALSE)
p <- p + theme(legend.position = c(0.89, 0.67),
               legend.background = element_blank(),
               legend.key = element_rect(fill = "white", color = NA),
               legend.title = element_blank(),
               axis.title = element_text(size = 10),
            #    text = element_text(size = 10),
               plot.tag.position = c(0.475, 0.95),
               plot.tag = element_text(size = 9.45, color = "#555555"))
p <- p + labs(tag = "Max.Orientation          (0.3, 0.1, 0.6)")
p <- p + geom_segment(
                x = 0.37, y = 0.75,
                xend = 0.25, yend = 0.63,
                lineend = "round", # See available arrow types in example above
                linejoin = "round",
                size = 0.4, 
                arrow = arrow(length = unit(0.1, "inches")),
                colour = "#EC7014" # Also accepts "red", "blue' etc
  )

ggsave('losses.pdf', height = 2.3, width = 3.2)