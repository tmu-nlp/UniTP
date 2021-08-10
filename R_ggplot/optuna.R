library(ggplot2)

en <- read.csv('optuna.disco.en.csv')
en$lang <- rep('DPTB - English', length(en$rank))
de <- read.csv('optuna.disco.de.csv')
de$lang <- rep('TIGER - German', length(de$rank))
data <- rbind(en, de)

p <- ggplot(data, aes(rank, score, facets = lang))
p <- p + geom_line()
p <- p + labs(x = 'Model Rank by Y Axis', y = 'F1 Score (Dev)')
p <- p + facet_wrap(~lang, scale = 'free')

p


ggsave('optuna.pdf', height = 1.7, width = 5.1)