source('data.R')
library(reshape2)
library(ggpubr)

orif_levels <- c('left', 'midin25', 'midin50', 'midin75', 'right', 'head')
orif_labels <- c('0\nleft', '0.25', '0.5', '0.75', '1\nright', 'head')

ratio_levels <- c('right_r', 'joint_r')
ratio_labels <- c('\'R\' of\norientation', 'positive\njoint')

corp_levels <- c('dptb', 'tiger')
corp_labels <- c('DPTB', 'Tiger')

medf_levels <- c('continuous', 'head', 'random', 'left', 'right')
medf_labels <- c('continuous', 'head', 'random', 'leftmost', 'rightmost')

load_lnr <- function(corp) {
    data <- load_corp('orif-LNR', corp)
    data$right_r <- data$right / (data$left + data$right)
    data$joint_r <- data$joint / (data$left + data$neutral + data$right)
    data$orif <- factor(data$orif, levels = orif_levels, labels = orif_labels)
    data
}

draw_orientation_joint <- function() {
    data <- rbind(load_lnr('dptb'), load_lnr('tiger'))
    data <- melt(data = data, id.vars = c('orif', 'corp'), measure.vars = ratio_levels)
    data$corp <- factor(data$corp, levels = c('dptb', 'tiger'), labels = c('DPTB', 'Tiger'))
    data$ratio <- factor(data$variable, levels = ratio_levels, labels = ratio_labels)
    p <- ggplot(data, aes(orif, value, color = ratio, shape = ratio))
    p <- p + facet_wrap(~corp)
    p <- p + geom_point()
    p <- p + theme(strip.text.x = element_text(margin = margin(b = 1.2, t = 0.1)),
                   axis.title.x = element_blank(),
                   axis.title.y = element_blank(),
                   legend.position = "none")
    #                legend.key.height = unit(1.8, "line"))
    # leg <- 'Signal\nPercentage'
    # p <- p + labs(color = leg, shape = leg)
    p

    ggsave(paste0(folder, 'disco.RJ.pdf'), height = 1.4, width = 4)
}

load_medf <- function(corp) {load_corp('sent.medf-node', corp)}
load_orif <- function(corp) {load_corp('sent.orif-node', corp)}
load_layer <- function(corp) {load_corp('compress.ratio.matrix', corp)}

limit_fn <- function(x) {
    low <- min(x)
    high <- max(x)
    c(low - .0012, high + .0012)
}

draw_m_signals <- function() {
    data <- rbind(load_layer('dptb'), load_layer('tiger'))
    data_summary <- data.frame()
    for (corp_i in corp_levels) {
        for (medf_i in medf_levels) {
            sub_data <- subset(data, medf == medf_i & corp == corp_i)
            lin_sum <- sum(sub_data$len)
            disco.1d <- sum(sub_data$n_comp) / lin_sum
            disco.2d <- sum(sub_data$n_positive) / sum(sub_data$n_child ** 2)
            joint <- 1 - sum(sub_data$n_chunk) / (lin_sum + 1)
            sub_data <- data.frame(value = disco.1d, type = 'disco.1d', medf = medf_i, corp = corp_i)
            data_summary <- rbind(data_summary, sub_data)
            sub_data <- data.frame(value = disco.2d, type = 'disco.2d', medf = medf_i, corp = corp_i)
            data_summary <- rbind(data_summary, sub_data)
            sub_data <- data.frame(value = joint, type = 'joint', medf = medf_i, corp = corp_i)
            data_summary <- rbind(data_summary, sub_data)
        }
    }
    print(data_summary)
    data_summary$corp <- factor(data_summary$corp, levels = corp_levels, labels = corp_labels)
    data_summary$medf <- factor(data_summary$medf, levels = medf_levels, labels = medf_labels)
    data_summary$type <- factor(data_summary$type, levels = c('disco.2d', 'joint', 'disco.1d'), labels = c('2D', 'J', '1D'))

    p <- ggplot(data_summary, aes(medf, value, color = type, shape = type))
    p <- p + facet_grid(type~corp, scale = 'free_y')
    p <- p + scale_y_continuous(limits = limit_fn)
    p <- p + geom_point()
    p <- p + theme(strip.text.x = element_text(margin = margin(b = 1.3, t = 0.1)),
                   strip.text.y = element_blank(),#element_text(margin = margin(l = 1.3, r = 0.1)),# family = 'Times', face = 'italic'),
                   axis.title.x = element_blank(),
                   axis.title.y = element_blank(),
                   panel.grid.minor.y = element_blank(),
                   panel.spacing.y = unit(0.4, "lines"),
                   legend.position = "none",
                   axis.text.x = element_text(angle = 18, hjust = 0.8))
    ggsave(paste0(folder, 'disco.DS.pdf'), height = 2.3, width = 4)
}

# draw_orientation_joint()
# draw_m_signals()


draw_lm <- function() {
    db <- rbind(load_orif('dptb'), load_orif('tiger'))
    dm <- rbind(load_medf('dptb'), load_medf('tiger'))
    data <- data.frame()

    for (corp_i in corp_levels) {
        for (orif_i in orif_levels) {
            sub_data <- subset(db, orif == orif_i & corp == corp_i)
            d_lm <- lm(size ~ len + I(len^2), data = sub_data)
            # lm_3 <- lm(size ~ len + I(len^2) + I(len^3), data = sub_data)$coefficients[4]
            sub_data <- data.frame(corp = corp_i, model = 'DB', factor = orif_i)
            for (coeff_i in seq(2)) {
                sub_data$coeff <- coeff_i
                sub_data$value <- d_lm$coefficients[coeff_i + 1]
                data <- rbind(data, sub_data)
            }
            # sub_data$coeff <- 3
            # sub_data$value <- lm_3
            # data <- rbind(data, sub_data)
        }

        for (medf_i in medf_levels) {
            sub_data <- subset(dm, medf == medf_i & corp == corp_i)
            sub_data$combined <- sub_data$linear + sub_data$square
            d_lm <- lm(combined ~ len + I(len^2), data = sub_data)
            # lm_3 <- lm(combined ~ len + I(len^2) + I(len^3), data = sub_data)$coefficients[4]
            sub_data <- data.frame(corp = corp_i, model = 'DM', factor = medf_i)
            for (coeff_i in seq(2)) {
                sub_data$coeff <- coeff_i
                sub_data$value <- d_lm$coefficients[coeff_i + 1]
                data <- rbind(data, sub_data)
            }
            # sub_data$coeff <- 3
            # sub_data$value <- lm_3
            # data <- rbind(data, sub_data)
        }
    }
    data$corp <- factor(data$corp, levels = corp_levels, labels = corp_labels)
    data$model <- factor(data$model)
    data$coeff <- factor(data$coeff, levels = seq(2), labels = c('Linear Coeff.', 'Quadratic Coeff.'))
    print(data)
    p <- ggplot(data, aes(corp, value, color = factor, shape = model))
    p <- p + facet_wrap(~coeff, scale = 'free_y')
    p <- p + geom_jitter(height = 0)
    p <- p + guides(shape = guide_legend(title = "Model"), color = 'none')
    p <- p + theme(strip.text.x = element_text(margin = margin(b = 1.2, t = 0.1)),
                   axis.title.x = element_blank(),
                   axis.title.y = element_blank())
    p
    ggsave(paste0(folder, 'disco.lm.pdf'), height = 1.3, width = 4)
    
    # data$medf <- factor(data$medf, levels = medf_levels, labels = medf_labels)
    # #len,height,gap,linear,square_layers,square,max_square,medf

    # p <- ggplot(data, aes(len + .5, linear + square + .5))
    # p <- p + labs(x = 'Sentence Length', y = 'Linear + Square Nodes')
    # p <- p + theme(strip.text.x = element_text(margin = margin(b = 1.2, t = 0.1)),
    #                strip.text.y = element_text(margin = margin(l = 1.2, r = 0.1))
    #                )
    # p <- p + coord_cartesian(xlim = c(NA, 95), ylim = c(NA, 500))
    # p <- p + geom_bin2d(binwidth = c(1, 1))#, aes(fill = log(..count..)))
    # p <- p + scale_fill_continuous(low = "darkblue", high = "tomato")
    # p <- p + theme(legend.position = "none")
    # p <- p + stat_smooth(method = 'lm', formula = y ~ x + I(x^2), geom = 'line', alpha = 0.5, color = 'black', show.legend=FALSE)

    # p

}

# draw_lm()

matrix_density <- function(corp_name) {
    db <- load_orif(corp_name)
    dm <- load_medf(corp_name)
    # print(head(db$height))
    # print(head(dm))
    print(mean(db$len)/ mean(dm$len))
    print(mean(db$height) / mean(dm$height))
    data <- load_layer(corp_name)
    # data <- subset(data, medf == 'head')
    print(sum(data$n_comp > 0) / nrow(data))
}

matrix_density('dptb')
matrix_density('tiger')

# draw_height_to_comp <- function() {
#     data <- rbind(load_layer('dptb'), load_layer('tiger'))
#     data$corp <- factor(data$corp, levels = corp_levels, labels = corp_labels)
#     data$medf <- factor(data$medf, levels = medf_levels, labels = medf_labels)
#     #len,height,ratio,n_comp,n_child,n_positive,medf

#     p <- ggplot(data, aes(height + .5, n_comp + .5))
#     p <- p + facet_grid(corp~medf)#, scale = 'free_y')
#     p <- p + theme(legend.position = "none",
#                    strip.text.x = element_text(margin = margin(b = 1.2, t = 0.1)),
#                    strip.text.y = element_text(margin = margin(l = 1.2, r = 0.1))
#                    )
#     p <- p + geom_bin2d(binwidth = c(1, 1), aes(fill = log(..count..)))
#     p <- p + scale_fill_continuous(low = "blue", high = "tomato")
#     p <- p + labs(x = 'Tree Height', y = '# of Disc. Constituents   ')
#     p <- p + scale_y_continuous(breaks = c(0, 2, 4, 6, 8))
#     p
#     ggsave(paste0(folder, 'disco.n_comp.pdf'), height = 1.8, width = 4.5)
# }

# draw_m_complexity <- function() {
#     draw_len_to_square()
#     draw_height_to_comp()   
# }


# draw_len_to_square()
# draw_m_complexity()








    # box <- melt(data = data, id.vars = c('corp'), measure.vars = c('gap', 'height'))
    # p <- ggplot(box, aes(corp, value))
    # p <- p + geom_violin()
    # p <- p + facet_wrap(~variable, scale = 'free_y')
    # p_gap <- p

    # p <- ggplot(data, aes(height + .5, gap + .5))
    # p <- p + facet_wrap(~corp)
    # p <- p + labs(x = 'Tree Height', y = 'Gap Degree')
    # p <- p + theme(strip.text.x = element_blank(),
    #                strip.background = element_blank(),)
    # # p <- p + facet_grid(corp~medf)
    # p_gap <- common_sent(p)

    # p <- ggarrange(p_lin, p_gap, ncol=1, heights = c(0.7, 0.9))
    # p <- annotate_figure(p,
    #                      bottom = text_grob('', size = 70),
    #                      left = text_grob("Compress Ratio", rot = 90, size = 70))