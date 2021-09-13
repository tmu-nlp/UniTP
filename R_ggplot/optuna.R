library(ggplot2)

task <- 'xccp'

en <- read.csv(paste0('optuna.', task, '.en.csv'))
en$lang <- rep('DPTB - English', length(en$rank))
de <- read.csv(paste0('optuna.', task, '.de.csv'))
de$lang <- rep('TIGER - German', length(de$rank))
data <- rbind(en, de)

if (task == 'dccp') {
    factor_labels <- c('Left', '25%', '50%', '75%', 'Right', 'Head')
    factor_levels <- c('left', 'midin25', 'midin50', 'midin75', 'right', 'head')
} else {
    factor_labels <- c('Left', 'Rand', 'Right')
    factor_levels <- c('left', 'random', 'right')
}
data$orif <- factor(data$orif, levels = factor_levels, labels = factor_labels)

p <- ggplot(data, aes(rank, score, color = orif, facets = lang))
p <- p + geom_point()
p <- p + labs(x = 'Model Rank by Y Axis', y = 'F1 Score (Dev)')
p <- p + facet_wrap(~lang, scale = 'free')
p <- p + theme(legend.title = element_blank())

p

ggsave(paste0(task, '.pdf'), height = 1.7, width = 5.1)