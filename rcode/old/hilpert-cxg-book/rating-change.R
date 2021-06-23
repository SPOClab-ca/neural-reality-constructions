library(ggplot2)
library(tidyverse)

data <- read_csv("rating-change.csv") %>%
  filter(Construction == "WAY")

ggplot(data, aes(Diff)) +
  geom_histogram(col = "white",
                 binwidth = 1)+
  scale_x_continuous(breaks=seq(-5, 5), limits=c(-5, 5)) +
  xlab("Δ Rating") +
  ylab("Count") +
  theme_bw()
  
