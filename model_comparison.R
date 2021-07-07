
setwd('C:/Users/Yair/OneDrive/Desktop/GA_course/Submissions/Projects/project_03/model_scores')
#setwd('C:/Users/Yair/OneDrive/Desktop/GA_course/Submissions/Projects/project_03/model_scores/model_scores_subset')

files <- list.files(pattern = '.csv')
files

f <- read.csv(paste(files[1]), header=T)

for (i in 2:length(files)){
  file <- read.csv(paste(files[i]), header=T)
  f <- rbind(f, file)
}

f$score <- as.factor(f$score)
levels(f$score) <- c("accuracy", 'f1', 'misclassification', 'precision', 'sensitivity', 'specificity')
f$score <- factor(f$score, levels=c("accuracy", 'misclassification', 'sensitivity', 'specificity', 'precision', 'f1'))

f$vectorizer <- as.factor(f$vectorizer)
levels(f$vectorizer) <- c('Count Vectorizer', 
                          'TF-IDF Vectorizer')

f$model <- as.factor(f$model)
levels(f$model) <- c('Logistic Regression','Naive Bayes', 'Random Forest')

library(ggplot2)
ggplot(subset(f,score!='misclassification'), 
       aes(y=value, x=score, group=vectorizer, fill=vectorizer,
              color=vectorizer)) + 
  theme_bw() + facet_grid(.~model)+
  geom_bar(stat='identity', position='dodge') +
  ylab('Score value') + 
  xlab('Score') + 
  scale_color_manual(values = c("#E69F00", "#56B4E9")) +
  scale_fill_manual(values = c("#E69F00", "#56B4E9")) +
  geom_hline(yintercept=0.5, color='darkred', linetype=2, size=2) + 
  scale_y_continuous(breaks=c(0,0.2,0.4,0.6,0.8,1)) +
  ggtitle('Different metrics for the different models and text vectorizers') + 
  theme(
    axis.title = element_text(size=22, color='black'),
    axis.text = element_text(size=20, color='black'),
    strip.text = element_text(size=22, color='black'),
    axis.text.x = element_text(angle = 35),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.text = element_text(size=20, color='black'),
    legend.title = element_text(size=20, color='black'),
    title = element_text(size=24, color='black')
  )
