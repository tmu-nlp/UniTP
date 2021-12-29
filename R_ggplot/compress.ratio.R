library(ggpubr)
source('data.R')

common_p <- function(p, max_x, text_x, text_y) {
    p <- p + facet_wrap(~orif, ncol = 1)
    p <- p + geom_point()
    p <- p + annotate("text", x = 2, y = 0.532, label = "Top", size = 14)
    breaks <- c(2, 50)
    while (tail(breaks, n = 1) + 80 < max_x) {
        breaks <- c(breaks, tail(breaks, n = 1) + 50)
    }
    p <- p + scale_x_continuous(breaks = c(breaks, max_x))
    p <- p + scale_size_continuous(range = c(1, 30))
    p <- p + scale_color_continuous(low = "blue", high = "tomato")
    # p <- p + stat_smooth(geom = 'line', alpha = 0.5, color = 'black', show.legend = FALSE, size = 2.5, method = 'rlm')#, n = 100)
    p <- p + stat_smooth(method = 'glm', formula = y ~ splines::bs(x, 23), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE, size = 2.5, n = 300)
    # p <- p + labs(tag = unname(TeX('$\\leftarrow$ Bottom-up combinatory direction')))
    p <- p + annotate('text', x = 180, y = 0.83, label = unname(TeX('$\\leftarrow$ Bottom-up combinatory direction')), size = 14)
    p <- p + annotate("text", x = text_x, y = text_y, label = unname(TeX("|$\\leftarrow$ Bottom Layers from Long Sentences $\\rightarrow$|")), size = 15, color = 'gray')
    p
}

draw <- function(corp, factor_levels, factor_labels) {
    orifs <- get_orifs(factor_levels, factor_labels)
    fname <- corp
    corp <- tolower(corp)
    data <- read.csv(paste0(folder, 'compress.ratio.', corp, '.csv'))
    max_x <- max(data$size)
    text_x <- (max_x - 89) * (249 / max_x)
    multi <- subset(data, orif == 'multi')
    multi$orif <- factor(multi$orif, levels = c('multi'), labels = c('Multi-branching'))
    multi <- multi[order(-multi$count),]
    data <- subset(data, orif != 'multi')
    data$orif <- factor(data$orif, levels = factor_levels, labels = factor_labels)
    data <- data[order(-data$count),]

    p <- ggplot(data, aes(size, ratio, size = count, color = log(count), weight = count, alpha = I(0.5)))
    p <- p + coord_cartesian(xlim = c(2, NA), ylim = c(0.45, 0.98))
    p <- p + theme(legend.position = "none",
                   axis.title = element_blank(),
                   axis.text.x = element_blank(),
                   axis.text.y = element_text(size = 45),
                   panel.grid.minor.y = element_blank(),
                   text = element_text(size = 70),
                   plot.tag = element_text(size = 40))
    p <- p + scale_y_continuous(breaks = c(1/2, 2/3, 3/4, 7/8, 1), labels = c('1/2', '2/3', '3/4', '7/8', '1'))
    p <- common_p(p, max_x, text_x, 0.5)
    

    p_ <- ggplot(multi, aes(size, ratio, size = count, color = log(count), weight = count, alpha = I(0.5)))
    p_ <- p_ + coord_cartesian(xlim = c(2, NA), ylim = c(0.04, 0.96))
    p_ <- p_ + facet_wrap(~orif, ncol = 1)
    p_ <- p_ + theme(legend.position = "none",
                     axis.title = element_blank(),
                     axis.text.y = element_text(size = 45),
                     panel.grid.minor.y = element_blank(),
                     text = element_text(size = 70),
                     #    plot.tag.position = c(0.73, 0.7),
                     plot.tag = element_text(size = 40))
    p_ <- p_ + scale_y_continuous(breaks = c(0, 1/4, 1/2, 3/4, 1), labels = c(0, '1/4', '1/2', '3/4', '1'))
    p_ <- common_p(p_, max_x, text_x, 0.11)

    p_num <- length(factor_levels)
    p <- ggarrange(p, p_, ncol=1, heights = c(0.25 * p_num, 0.3))
    annotate_figure(p,
                    bottom = text_grob(paste(fname, 'Layer Size'), size = 70),
                    left = text_grob("Compress Ratio", rot = 90, size = 70))
    ggsave(paste0(folder, 'compress.ratio.', corp,'.pdf'), height = 5 * (p_num + 1), width = 20)
}

# ptb /249; ptb /240; ptb /254
factor_levels <- c('left', 'right', 'midin', 'midout')
factor_labels <- c('CNF Left', 'CNF Right', 'Non-CNF Midin', 'Non-CNF Midout')
draw('PTB', factor_levels, factor_labels)
draw('CTB', factor_levels, factor_labels)
draw('KTB', factor_levels, factor_labels)

factor_levels <- c('left', 'midin25', 'midin50', 'midin75', 'right', 'head')
factor_labels <- paste0('rho = ', c(0, 0.25, 0.5, 0.75, 1, 'head'))
draw('DPTB', factor_levels, factor_labels)
draw('TIGER', factor_levels, factor_labels)