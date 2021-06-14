library(tidyverse)
library(ggplot2)

overall_df <- read_csv("./short_fn_perturbations.csv")
ratings_df <- read_csv("./short_fn_ratings.csv")

# Join two CSVs
overall_df$original_rating <-
  overall_df %>%
  merge(ratings_df, by.x="original_sent", by.y="text") %>%
  pull(score)
overall_df$replaced_rating <-
  overall_df %>%
  merge(ratings_df, by.x="replaced_sent", by.y="text") %>%
  pull(score)

# Does perturbation usually decrease score?
overall_df$delta <- overall_df$replaced_rating - overall_df$original_rating
ggplot(overall_df, aes(delta)) +
  geom_histogram(col = "white",
                 binwidth = 1)+
  scale_x_continuous(breaks=seq(-5, 5), limits=c(-5.5, 5.5)) +
  xlab("Δ Rating") +
  ylab("Count") +
  theme_bw()

# Overlapping histograms of originals and perturbed
overall_df %>%
  select(original_rating, replaced_rating) %>%
  gather('type') %>%
  ggplot(aes(x=value, fill=type)) +
  geom_bar(pos="dodge") +
  theme_bw()

