source('utils.R')
library(reshape2)
library(ggpubr)
library(scales)
folder <- 'stat.model/'

load_optuna <- function(model_name, corp_name) {
    fname <- paste('optuna', model_name, sep = '.')
    data <- load_corp(fname, corp_name)
    # data <- subset(data, lr > 10e-6)
    # min_tf <- min(data$tf) + 0.01
    # data <- subset(data, tf > min_tf)
    data
}

load_db <- function(model_name, corp_name) {
    #,lr,tf,df
    levels <- c('l.tag', 'l.label', 'l.joint', 'l.orient', 'l.shuffle',
                'b.left','b.midin25', 'b.midin50', 'b.midin75', 'b.right')
    data <- melt(data = load_optuna(model_name, corp_name),
                 id.vars = c('tf', 'df'),
                 measure.vars = c(levels, 'lr'))
    ns <- subset(data, variable != 'lr')
    ls <- subset(data, variable == 'lr')
    ns$variable <- factor(ns$variable, levels = levels,
                          labels = c('Tag', 'Label', 'Joint\nDB Losses', 'Orient ', ' Shuffle', 0, 0.25, '0.5\nBinarization', 0.75, 1))
    ls$variable <- factor(ls$variable, levels = c('lr'), labels = c('Learning Rate'))
    list(ns = ns, ls = ls)
}

load_dm <- function(model_name, corp_name) {
    #, 'm.head', 'm.continuous', 'm.left', 'm.right'
    data <- melt(data = load_optuna(model_name, corp_name),
                 id.vars = c('tf', 'df'),
                 #  measure.vars = c('lr'))
                 measure.vars = c('l.tag', 'l.label', 'l.joint', 'l.disc', 'l.biaff', 'l.neg',
                                  'm.neg', 'm.sub', 'lr'))#, 'm.left', 'm.random', 'm.right'))
    ns <- subset(data, variable != 'lr' & variable != 'm.neg')
    ls <- subset(data, variable == 'lr' | variable == 'm.neg')
    ns$variable <- factor(ns$variable, levels = c('l.tag', 'l.label', 'l.joint', 'l.disc', 'l.biaff', 'l.neg', 'm.sub'),
                          labels = c('Tag', 'Label', 'Joint', 'Disc.\nDM Losses', ' Biaff.', 'nBiaff.', '   P(sub)'))
    ls$variable <- factor(ls$variable, levels = c('m.neg', 'lr'), labels = c('P(neg)  \nSampling Rates                        ', '                  Learning Rate'))
    list(ns = ns, ls = ls)
}

break_fn <- function(x) {
    x_max <- max(x)
    x_min <- min(x)
    x_len <- x_max - x_min
    x_1 <- round((x_len) / 3 + x_min, digits=2)
    x_2 <- round((x_len) / 3 * 2 + x_min, digits=2)
    c(x_min, x_1, x_2, x_max)
}

draw <- function(data) {
    data <- data[order(data$tf), ]

    p <- ggplot(data, aes(variable, value, color = tf, size = tf))
    # p <- p + facet_wrap(~coeff, scale = 'free_y')
    p <- p + geom_jitter(width = 0.0, height = 0, alpha = I(0.9)) #+ geom_violin()
    p <- p + guides(size = guide_legend(title = "F1"), color = guide_legend(title = "F1"))
    p <- p + scale_size_continuous(range = c(.1, 5), breaks = break_fn)
    p <- p + scale_color_continuous(low = "blue", high = "tomato", breaks = break_fn)
    p <- p + theme(#strip.text.x = element_text(margin = margin(b = 1.2, t = 0.1)),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    legend.margin = margin(l = 4, r = 4, b = 20),
                    legend.background = element_rect(fill='transparent'))
    p
}

proc <- function(model_name, corp_name) {
    if (model_name == 'db') {
        data <- load_db(model_name, corp_name)
        ws <- c(5, 1)
    } else {
        data <- load_dm(model_name, corp_name)
        ws <- c(5.5, 2)
    }
    np <- draw(data$ns)
    lp <- draw(data$ls)
    lp <- lp + scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
              labels = trans_format("log10", math_format(10^.x)))
    ggarrange(np, lp, ncol = 2, widths = ws, common.legend = TRUE, legend = 'right')
    fname <- paste('optuna', model_name, corp_name, 'pdf', sep = '.')
    ggsave(paste0(folder, fname), height = 1.7, width = 5.6)
}

proc('db', 'tiger')
proc('db', 'dptb')
proc('dm', 'tiger')
proc('dm', 'dptb')
# en <- read.csv(paste0('optuna.', task, '.en.csv'))
# en$lang <- rep('DPTB - English', length(en$rank))
# de <- read.csv(paste0('optuna.', task, '.de.csv'))
# de$lang <- rep('TIGER - German', length(de$rank))
# data <- rbind(en, de)

# if (task == 'dccp') {
#     factor_labels <- c('Left', '25%', '50%', '75%', 'Right', 'Head')
#     factor_levels <- c('left', 'midin25', 'midin50', 'midin75', 'right', 'head')
# } else {
#     factor_labels <- c('Left', 'Rand', 'Right')
#     factor_levels <- c('left', 'random', 'right')
# }
# data$orif <- factor(data$orif, levels = factor_levels, labels = factor_labels)

# p <- ggplot(data, aes(rank, score, color = orif, facets = lang))
# p <- p + geom_point()
# p <- p + labs(x = 'Model Rank by Y Axis', y = 'F1 Score (Dev)')
# p <- p + facet_wrap(~lang, scale = 'free')
# p <- p + theme(legend.title = element_blank())

# p
# ggsave(paste0(task, '.pdf'), height = 1.7, width = 5.1)