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
    # data <- subset(data, gain > 0)
    data
}

load_db_dis <- function(model_name, corp_name) {
    levels <- c('tag', 'label', 'joint', 'orient', 'shuffled_joint', 'shuffled_orient', 'left','midin25', 'midin50', 'midin75', 'right')
    labels <- c('Tag', 'Label', 'Joint\nDB Loss Weights    ', ' Orient ', 'Sff\nJoint', 'Sff\n Orient', '0\n   Left', 0.25, '0.5\nBinarization', 0.75, '1\nRight      ')
    data <- melt(data = load_optuna(model_name, corp_name),
                 id.vars = c('dev.tf', 'tf', 'df'),
                 measure.vars = c(levels, 'lr'))
    ns <- subset(data, variable != 'lr')
    ls <- subset(data, variable == 'lr')
    ns$variable <- factor(ns$variable, levels = levels, labels = labels)
    ls$variable <- factor(ls$variable, levels = c('lr'), labels = c('\nLearning Rate              '))
    list(ns = ns, ls = ls)
}

beta_to_data <- function(parameters, n) {
    gx <- c()
    gy <- c()
    gg <- c()
    gs <- c()
    gt <- c()
    for (i in seq_len(nrow(parameters))) {
        left <- parameters$left[i]
        right <- parameters$right[i]
        by <- 1 / n
        x <- seq(0, 1, by)
        y <- dbeta(x, left, right)
        # v <- y != Inf
        # x <- x[v]
        # y <- y[v]
        if (left > 1) {
            if (right > 1) {
                t <- 'b'
            } else {
                t <- 'l'
            }
        } else if (right > 1) {
            t <- 'r'
        } else {
            t <- 'n'
        }
        gx <- c(gx, x)
        gy <- c(gy, y)
        gg <- c(gg, rep(i, length(x)))
        gs <- c(gs, rep(parameters$dev.tf[i], length(x)))
        gt <- c(gt, rep(t, length(x)))
    }
    gg <- factor(gg)
    gt <- factor(gt, levels = c('b', 'l', 'r', 'n'), labels = c('Both', 'Left', 'Right', 'None'))
    data.frame(x = gx, y = gy, g = gg, tf = gs, gt = gt)
}

load_db_con <- function(model_name, corp_name) {
    levels <- c('tag', 'label', 'joint', 'orient', 'shuffled_joint', 'shuffled_orient', 'msb')
    con <- c('left', 'right')
    data <- load_optuna(model_name, corp_name)
    beta <- beta_to_data(data, 500)
    data <- melt(data = data,
                 id.vars = c('dev.tf', 'tf', 'df'),
                 measure.vars = c(levels, 'lr', con))
    ns <- subset(data, variable %in% levels)
    ls <- subset(data, variable == 'lr')
    bs <- subset(data, variable %in% con)
    nl <- scale_x_discrete(labels = c(
        'tag' = parse(text = TeX('$\\alpha_{tag}$')),
        'label' = parse(text = TeX('$\\alpha_{label}$')),
        'joint' = parse(text = TeX('$\\alpha_{jnt}$')),
        'orient' = parse(text = TeX('$\\alpha_{ori}$')),
        'shuffled_joint' = parse(text = TeX('$\\alpha^{sff}_{jnt}$')),
        'shuffled_orient' = parse(text = TeX('$\\alpha^{sff}_{ori}$')),
        'msb' =  expression(paste(rho, symbol("\306")))))
    ll <- scale_x_discrete(labels = c('lr' = parse(text = TeX('$\\gamma$'))))
    bl <- scale_x_discrete(labels = c(
        'left' = parse(text = TeX('$\\alpha_{left}$')),
        'right' = parse(text = TeX('$\\alpha_{right}$'))))
    ns$variable <- factor(ns$variable, levels = levels)
    ls$variable <- factor(ls$variable, levels = c('lr'))
    bs$variable <- factor(bs$variable, levels = con)
    list(ns = ns, ls = ls, nl = nl, ll = ll, bs = bs, bl = bl, beta = beta)
}

load_dm_ext <- function(model_name, corp_name) {
    data <- load_optuna(model_name, corp_name)
    data$max_inter_height <- data$max_inter_height / max(data$max_inter_height)
    levels <- c('tag', 'label', 'fence', 'disco_1d', 'disco_2d', 'disco_2d_intra', 'disco_2d_inter', 'msb')
    log.levels <- c('intra_rate', 'inter_rate', 'lr')
    data <- melt(data = data,
                 id.vars = c('dev.tf', 'tf', 'df'),
                 measure.vars = c(levels, log.levels))
    ls <- data$variable %in% log.levels
    ns <- subset(data, !ls)
    ls <- subset(data, ls)
    nl <- scale_x_discrete(labels = c(
        'tag' = parse(text = TeX('$\\beta_{tag}$')),
        'label' = parse(text = TeX('$\\beta_{label}$')),
        'fence' = parse(text = TeX('$\\beta_{jnt}$')),
        'disco_1d' = parse(text = TeX('$\\beta_{disc}$')),
        'disco_2d' = parse(text = TeX('$\\beta_{D}$')),
        'disco_2d_intra' = parse(text = TeX('$\\beta_{C}$')),
        'disco_2d_inter' = parse(text = TeX('$\\beta_{X}$')),
        'msb' =  expression(paste(rho, symbol("\306")))))
    ll <- scale_x_discrete(labels = c(
        'intra_rate' = parse(text = TeX('$\\beta_{c}$')),
        'inter_rate' = parse(text = TeX('$\\beta_{x}$')),
        'lr' = parse(text = TeX('$\\gamma$'))))
    ns$variable <- factor(ns$variable, levels = levels)
    ls$variable <- factor(ls$variable, levels = log.levels)
    list(ns = ns, ls = ls, nl = nl, ll = ll)
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
    data <- data[order(data$dev.tf), ]

    p <- ggplot(data, aes(variable, value, color = dev.tf, size = dev.tf))
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
        data <- load_db_con(model_name, corp_name)
        beta <- data$beta
        if (!is.null(beta)) {
            beta <- beta[order(beta$tf), ]
            p <- ggplot(beta, aes(x, y, group = g, alpha = tf, color = gt, size = tf))
            p <- p + geom_line()
            gf <- guide_legend(title = '\nDev F1', order = 1, reverse = TRUE)
            gb <- guide_legend(title = unname(TeX('$\\alpha_l,\\,\\alpha_r\\,>\\,1$')), order = 2)
            p <- p + guides(alpha = gf, size = gf, color = gb)
            p <- p + scale_size_continuous(range = c(0.2, 1.2))
            p <- p + coord_cartesian(ylim = c(0, 16.6))
            p <- p + theme(
                axis.title = element_blank(),
                axis.text.y = element_blank(),
                axis.ticks.y = element_blank(),
                legend.spacing.y = unit(0.0, 'cm'))
            if (corp_name == 'tiger') {
                p <- p + annotate("text",
                    label = "Probability density function (PDF)\n\nBest:   Beta(8.4, 10.9)",
                    x = 0.5, y = 13, size = 4.5)
                p <- p + annotate("segment",
                    x = 0.5, y = 8,
                    xend = 0.43, yend = 3.6,
                    lineend = "round",
                    linejoin = "round",
                    arrow = arrow(length = unit(0.1, "inches")))
            }
            p
            ggsave(paste0(folder, paste('bin', model_name, corp_name, 'pdf', sep = '.')), height = 2.3, width = 5.5)
        }
        ws <- c(3.5, 1.5, 1)
        legend.position <- 'left'
        hc <- "#FFFF49"
        lc <- "#268D7F"
    } else {
        data <- load_dm_ext(model_name, corp_name)
        ws <- c(5.2, 2.3)
        legend.position <- 'right'
        hc <- "#5CCDFF"
        lc <- "#FF68A1"
    }
    yl <- scale_y_log10(
        breaks = trans_breaks("log10", function(x) 10^x),
        labels = trans_format("log10", math_format(10^.x)))
    gd <- guide_legend(title = paste0('Model ', toupper(model_name), '\nF1 | ', replace_corp_name(corp_name)), reverse = TRUE)
    np <- draw(data$ns, gd, lc, hc)
    np <- np + scale_y_continuous(breaks = seq(0, 1, 0.25), labels = c(0, '¼', '½', '¾', 1))
    np <- np + data$nl
    lp <- draw(data$ls, gd, lc, hc)
    lp <- lp + yl
    lp <- lp + data$ll
    if (is.null(data$bs)) {
        ggarrange(np, lp, ncol = 2, widths = ws, common.legend = TRUE, legend = legend.position)
    } else {
        bp <- draw(data$bs, gd, lc, hc) + data$bl + yl
        ggarrange(np, bp, lp, ncol = 3, widths = ws, common.legend = TRUE, legend = legend.position)
    }
    fname <- paste('optuna', model_name, corp_name, 'pdf', sep = '.')
    ggsave(paste0(folder, fname), height = 1.7, width = 5)
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

# score_plot()

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

load_2d <- function(sub) {
    if (sub) {
        data <- 'stat.2d.sub'
    } else {
        data <- 'stat.2d.nsb'
    }
    data <- rbind(load_corp(data, 'dptb'), load_corp(data, 'tiger'))
    data$sub <- rep(sub, nrow(data))
    # data$bin <- sapply(data$threshold, get.bin)
    for (c in c('dptb', 'tiger')) {
        d <- subset(data, corp == c & threshold != 0.5)
        n <- nrow(d)
        print(c)
        print(sub)
        print(sum(d$attempt) / n)
        print(nrow(subset(d, size == comp)))
    }
    # print(nrow(subset(data, threshold == 0.5)))
    # print(nrow(subset(data, threshold == -1 | threshold == 1)))
    # print(nrow(subset(data, size == comp)))
    # att <- subset(data, bin == 0 | bin > 1)$attempt
    # print(mean(att))
    # print(sd(att))
    data
}

tries_label_fn <- function(x) {
    y = c()
    for (i in x) {
        if (is.na(i) | i < 1.1) {
            y <- c(y, i)
        } else {
            y <- c(y, 'FAIL')
        }
    }
    y
}

disco_2d_error_plot <- function() {
    data <- rbind(load_2d(F), load_2d(T))
    data$corp <- factor(data$corp, levels = c('dptb', 'tiger'), labels = c('DPTB', 'Tiger'))
    data$sub <- factor(data$sub)
    levels(data$sub) <- c( # does not work: parse(text = TeX('$\\rho_{\\emptyset} = 0$'))
        'FALSE' = expression(paste(rho, symbol("\306"), " = 0")), 
        'TRUE'  = expression(paste(rho, symbol("\306"), ' > 0')))
    data$error <- data$threshold < 0
    data$threshold[data$error] <- 1.1
    error <- subset(data, error)
    lc <- '#141200'
    hc <- "#5CCDFF"

    p <- ggplot(data, aes(threshold, attempt, shape = error))# %/% size / size))
    # https://ggplot2-book.org/annotations.html
    # p <- p + geom_rect(aes(
    #     xmin = 0.9, xmax = 1.0,
    #     ymin = -Inf, ymax = Inf), fill = 'blue', alpha = 0.1)
    p <- p + geom_hline(yintercept = 50, alpha = I(0.31), color = '#1B6CBF', linetype = "dashed")
    p <- p + geom_count(aes(color = log(..n..)), alpha = I(0.8))
    p <- p + geom_count(data = error, aes(threshold, attempt, shape = error), color = "#FF68A1")
    p <- p + scale_x_continuous(
        breaks = c(-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1),
        labels = tries_label_fn)
    p <- p + scale_color_continuous(low = lc, high = hc)
    p <- p + labs(x = 'Threshold', y = '# Tries')
    # ggMarginal(p, type = "density")
    # p <- p + geom_xsidedensity(aes(y = after_stat(density)))
    # p <- p + geom_ysidedensity(aes(x = after_stat(density)))
    # p <- p + theme_ggside_void()
    p <- p + theme(
        legend.position = "none",
        strip.text.x = element_text(margin = margin(b = 1.8, t = 0.7)),
        strip.text.y = element_text(margin = margin(l = 1.8, r = 0.7)))
    p <- p + facet_grid(sub~corp, scale = 'free', space = 'free', labeller = label_parsed)
    # p <- p + geom_vline(xintercept = 1.05, alpha = I(0.2), color = 'red')
    # p <- p + geom_vline(xintercept = 0.9, alpha = I(0.2), color = 'red')
    p

    ggsave(paste0(folder, 'biaff.tries.pdf'), height = 3, width = 4)
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