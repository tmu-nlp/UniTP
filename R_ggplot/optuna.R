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
                          labels = c('Tag', 'Label', 'Joint \nDB Loss Weights', 'Orient ', '  Shuffle', '0\n   Left', 0.25, '0.5\nBinarization', 0.75, '1\nRight      '))
    ls$variable <- factor(ls$variable, levels = c('lr'), labels = c('\nLearning Rate              '))
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
                          labels = c('Tag', 'Label', 'Joint', 'Disc.\nDM Loss Weights', ' Biaff.', 'nBiaff.', '   P.sub'))
    ls$variable <- factor(ls$variable, levels = c('m.neg', 'lr'), labels = c('P.neg  \nSampling Rates                          ', '\n                  Learning Rate'))
    list(ns = ns, ls = ls)
}

load_dm_ext <- function(model_name, corp_name) {
    #, 'm.head', 'm.continuous', 'm.left', 'm.right'
    data <- load_optuna(model_name, corp_name)
    data$max_inter_height <- data$max_inter_height / max(data$max_inter_height)
    data <- melt(data = data, id.vars = c('tf', 'df'),
                 measure.vars = c('l.tag', 'l.label', 'l.joint', 'l.disc', 'l.biaff', 'l.disco_2d_intra', 'l.disco_2d_inter', 'max_inter_height',
                                  'r.intra', 'r.inter', 'r.sub', 'r.more_sub', 'lr'))
    ns <- subset(data, variable != 'lr' & variable != 'r.intra' & variable != 'r.inter')
    ls <- subset(data, variable == 'lr' | variable == 'r.intra' | variable == 'r.inter')
    ns$variable <- factor(ns$variable,
                          levels = c('l.tag', 'l.label', 'l.joint', 'l.disc', 'l.biaff', 'l.disco_2d_intra', 'l.disco_2d_inter', 'max_inter_height', 'r.sub', 'r.more_sub'),
                          labels = c('Tag', 'Label', 'Joint', 'Disc.\n                    DM Loss Weights (Intra  Inter)', ' Biaff.', 'L\'ba', 'L\'\'ba', 'Xi\n| [0,9]', '   P.sub', 'P.msb'))
    ls$variable <- factor(ls$variable, levels = c('r.intra', 'r.inter', 'lr'), labels = c('P\'ba         \n|    Sampling Rates                         ',
                                                                                          'P\'\'ba',
                                                                                          '|\n                  | Learning Rate'))
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

draw <- function(data, gd, lc, hc) {
    data <- data[order(data$tf), ]


    p <- ggplot(data, aes(variable, value, color = tf, size = tf))
    # p <- p + facet_wrap(~coeff, scale = 'free_y')
    p <- p + geom_point(alpha = I(0.618)) #+ geom_violin()
    p <- p + guides(size = gd, color = gd)
    p <- p + scale_size_continuous(range = c(1, 5), breaks = break_fn)
    p <- p + scale_color_continuous(low = lc, high = hc, breaks = break_fn)
    p <- p + theme(#strip.text.x = element_text(margin = margin(b = 1.2, t = 0.1)),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    legend.margin = margin(l = 4, r = 4, b = 20),
                    legend.background = element_rect(fill='transparent'))
    p
}

replace_corp_name <- function(x) {
    if(x == 'dptb') {
        return('DPTB')
    } else if (x == 'tiger') {
        return('Tiger')
    }
    x
}

hyperparameter_plot <- function(model_name, corp_name) {
    if (model_name == 'db') {
        data <- load_db(model_name, corp_name)
        ws <- c(5, 1)
        legend.position <- 'left'
        hc <- "tomato"
        lc <- "cyan"
    } else {
        data <- load_dm_ext(model_name, corp_name)
        ws <- c(5.5, 2)
        legend.position <- 'right'
        hc <- "yellow"
        lc <- "purple"
    }
    gd <- guide_legend(title = paste0('Model ', toupper(model_name), '\nF1 | ', replace_corp_name(corp_name)), reverse = TRUE)
    np <- draw(data$ns, gd, lc, hc)
    np <- np + scale_y_continuous(breaks = seq(0, 1, 0.25), labels = c(0, '¼', '½', '¾', 1))
    lp <- draw(data$ls, gd, lc, hc)
    lp <- lp + scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
              labels = trans_format("log10", math_format(10^.x)))
    ggarrange(np, lp, ncol = 2, widths = ws, common.legend = TRUE, legend = legend.position)
    fname <- paste('optuna', model_name, corp_name, 'pdf', sep = '.')
    ggsave(paste0(folder, fname), height = 1.7, width = 5.5)
}

hyperparameter_plot('db', 'tiger')
hyperparameter_plot('db', 'dptb')
hyperparameter_plot('dm', 'tiger')
hyperparameter_plot('dm', 'dptb')

load_scores <- function(model_name, corp_name) {
    fname <- paste('diff', 'optuna', model_name, sep = '.')
    data <- load_corp(fname, corp_name)
    data$model <- rep(model_name, nrow(data))
    data
}

score_corp <- function(corp_name) {
    variables <- c('dev.tf', 'test.df', 'n.step')
    data <- rbind(load_scores('db', corp_name), load_scores('dm', corp_name))
    data <- melt(data, # 'dev.tf', 'test.tf', 'test.df', 'n.step'
                 id.vars = c('test.tf', 'model'),
                 measure.vars = variables)
    # corp, x_axis grouped in c(dev.tf, n.step, test.df), y_axis(test.tf)
    data$model <- factor(data$model, levels = c('db', 'dm'), labels = c('DB', 'DM'))
    data$variable <- factor(data$variable, levels = variables, labels = c('Development F1', 'Test Disc. F1', 'BO Epoch'))
    print(head(data))
    p <- ggplot(data, aes(value, test.tf))
    p <- p + facet_grid(model~variable, scale = 'free', switch= 'x')
    p <- p + geom_count(aes(color = ..n..), alpha = I(0.618))
    p <- p + theme(axis.title = element_blank())
    p
}

score_plot <- function() {
    p_dptb <- score_corp('dptb')
    p_tiger <- score_corp('tiger')
    y_axis <- '            Tiger             Test F1                  DPTB'
    annotate_figure(ggarrange(p_dptb, p_tiger, ncol = 1, legend = 'none'), left = y_axis)
    ggsave(paste0(folder, 'optuna.score.pdf'), height = 4.5, width = 5)
}

score_plot()

get.bin <- function(threshold) {
    if (threshold == 0.5) {
        1
    } else if (threshold == -1 | threshold == 1 ) {
        7
    } else if (threshold < 0.5){
        0
    } else {
        floor(threshold * 10) - 3 # max = 6
    }
}

load_2d <- function(corp_name) {
    data <- load_corp('stat.2d.dm', corp_name)
    data$bin <- sapply(data$threshold, get.bin)
    # print(nrow(data))
    # print(nrow(subset(data, threshold == 0.5)))
    # print(nrow(subset(data, threshold == -1 | threshold == 1)))
    # print(nrow(subset(data, size == comp)))
    # att <- subset(data, bin == 0 | bin > 1)$attempt
    # print(mean(att))
    # print(sd(att))
    data
}

disco_2d_error_plot <- function() {
    data <- rbind(load_2d('dptb'), load_2d('tiger'))
    data$corp <- factor(data$corp, levels = c('dptb', 'tiger'), labels = c('DPTB - PLM DM', 'Tiger - PLM DM'))

    p <- ggplot(data, aes(bin, attempt))# %/% size / size))
    p <- p + geom_count(aes(color = ..n..), alpha = I(0.618))
    p <- p + scale_x_continuous(
        breaks = seq(0, 7),
        labels = c('<.5', '0.5', '<.6', '<.7', '<.8', '<.9', '<1', 'Err'))
    # p <- p + scale_y_sqrt()
    # p <- p + geom_rug(alpha = 0.05)
    p <- p + labs(x = 'Threshold Bin', y = '# Tries')
    # ggMarginal(p, type = "density")
    # p <- p + geom_xsidedensity(aes(y = after_stat(density)))
    # p <- p + geom_ysidedensity(aes(x = after_stat(density)))
    # p <- p + theme_ggside_void()
    p <- p + theme(legend.position = "none")
    p <- p + facet_wrap(.~corp, scale = 'free')
    p

    ggsave(paste0(folder, 'biaff.tries.pdf'), height = 1.7, width = 4.5)
}

disco_2d_error_plot()

# diff_dev_test <- function(data) {
#     p <- ggplot(data, aes(dev.tf, test.tf))
#     p <- p + labs(x = 'Development F1')
#     p
# }

# diff_step <- function(data) {
#     p <- ggplot(data, aes(n.step, test.tf))
#     p <- p + labs(x = 'Optuna Epoch')
#     p
# }

# diff_td_f1 <- function(data) {
#     p <- ggplot(data, aes(test.df, test.tf))
#     p <- p + labs(x = 'Test-set D.F1')
#     p
# }

plot_diff <- function(model_name, corp_name, ann_plot) {

    # numbers <- data$test.tf
    # numbers <- aggregate(numbers, list(num = numbers), length)
    # start.test.tf <- data[which.max(numbers$x),]$test.tf
    # data <- subset(data, test.tf >= start.test.tf)

    # dx <- density(data$test.tf)
    # dy <- approx(dx$x, dx$y, xout = data$test.tf)$y
    # od <- order(dy)
    # data <- data[od,]

    gd <- guide_legend(title = 'Count')
    gd <- guides(color = gd, size = gd)

    gc <- geom_count(aes(color = ..n..), alpha = I(0.618))
    gt <- theme(axis.title.y = element_blank(),
                axis.text.y  = element_blank(),
                axis.ticks.y = element_blank())
    g1 <- labs(y = 'Test-set F1')

    dt <- diff_dev_test(data) + gd + gc
    ns <- diff_step(data) + gd + gc
    f1 <- diff_td_f1(data) + gd + gc
    if (ann_plot) {
        tf_max <- max(data$test.tf)
        tf_bnd <- 83.92
        nf_y <- 84.038
        r_alpha <- .16
        t_alpha <- .32
        t_size <- 3
        a11 <- annotate("rect", xmin = 0, xmax = max(data$n.step), ymin = tf_bnd, ymax = tf_max, alpha = r_alpha)
        a12 <- annotate('text', label = 'w/ nBiaff.', x = 42, y = nf_y, size = t_size, alpha = t_alpha)
        a13 <- annotate('text', label = '^ optuna start', x = 15, y = 83.72, size = t_size, alpha = .618, color = 'dodgerblue')
        ns <- ns + a11 + a12 + a13

        a21 <- annotate("rect", xmin = min(data$dev.tf), xmax = max(data$dev.tf), ymin = tf_bnd, ymax = tf_max, alpha = r_alpha)
        a22 <- annotate('text', label = 'w/ nBiaff.', x = 88.48, y = nf_y, size = t_size, alpha = t_alpha)
        dt <- dt + a21 + a22
        
        df_min <- min(data$test.df)
        df_vex <- df_min + 1.62
        a31 <- annotate('text', label = 'Linearity: F1 & D.F1', x = 58.1, y = 84.05, size = t_size)
        a32 <- annotate("segment", x = df_min, xend = df_vex, y = tf_bnd, yend = tf_bnd, alpha = r_alpha, linetype = 2)
        a33 <- annotate("segment", x = df_vex, xend = df_vex, y = tf_bnd, yend = min(data$test.tf), alpha = r_alpha, linetype = 2)
        s3 <- stat_smooth(method = 'lm', formula = y ~ x, geom = 'line', color = 'black', show.legend = F)
        f1 <- f1 + a31 + a32 + a33 + s3
    }
    p <- ggarrange(ns + g1, dt + gt, f1 + gt, ncol = 3, widths = c(2.9, 2.3, 2.3), legend = 'none')
    # annotate_figure(p, top = toupper(paste(model_name, corp_name)))
    fname <- paste('diff', 'optuna', model_name, corp_name, 'pdf', sep = '.')
    ggsave(paste0(folder, fname), height = 1.9, width = 5.3)
}

# plot_diff('db', 'tiger', F)
# plot_diff('db', 'dptb', F)
# plot_diff('dm', 'tiger', F)
# plot_diff('dm', 'dptb', F)
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