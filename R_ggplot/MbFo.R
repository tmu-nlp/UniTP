source('utils.R')
library(ggpubr)

int_breaks <- function(x, n = 8) {
  l <- pretty(x, n)
  l[abs(l %% 1) < .Machine$double.eps ^ 0.5] 
}

folder <- 'stat.model/'

load_model_corp <- function(model, corp) {
    data <- load_corp(paste0('MbFo.', model), corp)
    data$model <- rep(model, nrow(data))
    data
}

plot_corp <- function(corp, first_row) {
    db <- load_model_corp('xbert_dccp', corp)
    dm <- load_model_corp('xbert_xccp', corp)
    data <- rbind(db, dm)
    data <- subset(data, metric == 'multib' & bin < 9 | metric == 'fanout' & bin < 4)
    data$metric <- factor(data$metric, levels = c('fanout', 'multib'), labels = c('Fan-out (k)', 'Multi-branching Arity (n-ary)'))
    data$corp <- factor(data$corp, levels = c('dptb', 'tiger'), labels = c('DPTB', 'Tiger'))
    data$model <- factor(data$model, levels = c('xbert_dccp', 'xbert_xccp'), labels = c('DB', 'DM'))
    # print(data)

    num_labels <- function(x) {
        if (length(x) > 4) {
            d <- subset(dm, metric == 'multib')$g
        } else {
            d <- subset(dm, metric == 'fanout')$g
        }
        s <- c()
        for(i in x) {
            if(is.na(i)) {
                s <- c(s, i)
            } else {
                s <- c(s, d[i])
            }
        }
        s
    }

    p <- ggplot(data, aes(bin, f1, group = model, color = model, shape = model))
    p <- p + facet_grid(corp~metric, scale = 'free', space = 'free_x',)# switch="x")
    p <- p + geom_line() + geom_point(size = 2)
    # p <- p + annotate("text", label = "DM-DB:", x = 7, y = subset(data, bin = 7)$f1[7], size = 1)

    fx <- element_text(margin = margin(b = 1.8, t = 0.7))
    fy <- element_text(margin = margin(l = 1.8, r = 0.7))
    if (first_row) {
        sec.name <- 'Number of Gold Test Samples'
        p <- p + theme(axis.text.x.bottom  = element_blank(),
                       axis.ticks.x.bottom = element_blank(),
                       legend.position = c(0.6, 0.2),
                       legend.background = element_blank(),
                       legend.direction = 'horizontal',
                       legend.title = element_text(size = 9),
                       strip.text.x = fx,
                       strip.text.y = fy,
                       plot.tag.position = c(0.02, 0.56),
                       plot.tag = element_text(size = 10))
        p <- p + labs(tag = "F1")
    } else {
        sec.name <- NULL
        p <- p + theme(strip.text.x = element_blank(),
                       strip.text.y = fy,
                       legend.position = "none")
    }
    p <- p + theme(axis.title.y = element_blank(),
                   axis.title.x.bottom = element_blank(),
                   plot.margin = unit(c(0,0,0,0),"cm"))

    gp <- guide_legend(title = "PLM Model")
    p <- p + guides(shape = gp, color = gp)
    p <- p + scale_x_continuous(breaks = int_breaks,
                                sec.axis = sec_axis(~.,
                                                    name = sec.name,
                                                    breaks = int_breaks,
                                                    labels = num_labels))
    p
}


p <- ggarrange(plot_corp('dptb', T), plot_corp('tiger', F), ncol=1, common.legend = F, heights = c(1, 0.9))
annotate_figure(p)

# p <- ggplot(data, aes(bin, f1, fill = model))
# p <- p + facet_grid(corp~metric, scale = 'free_x', space='free')
# p <- p + geom_bar(stat = 'identity', position = 'dodge')

ggsave(paste0(folder, 'MbFo.pdf'), height = 2.2, width = 4.1)