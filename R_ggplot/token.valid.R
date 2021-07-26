library(ggplot2)

s01 = read.csv('scores.00001.00.csv')
s01$epoch <- rep('1', length(s01$score))
# s02 = read.csv('scores.00002.00.csv')
# s02$epoch <- rep('2', length(s02$score))
# s05 = read.csv('scores.00005.00.csv')
# s05$epoch <- rep('5', length(s05$score))
# s10 = read.csv('scores.00010.00.csv')
# s10$epoch <- rep('10', length(s10$score))
# s20 = read.csv('scores.00020.00.csv')
# s20$epoch <- rep('20', length(s20$score))
# s50 = read.csv('scores.00050.00.csv')
# s50$epoch <- rep('50', length(s50$score))
s90 = read.csv('scores.00090.00.csv')
s90$epoch <- rep('90', length(s90$score))

data <- rbind(s01, s90)#, s05, s10, s20, s50)

p <- ggplot(data, aes(score, group = epoch, color = epoch))
p <- p + geom_density()
p


ggsave('token.valid.pdf', height = 2.7, width = 5.2, dpi = 600)