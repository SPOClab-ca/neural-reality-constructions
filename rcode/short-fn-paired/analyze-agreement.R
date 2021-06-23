library(tidyverse)
library(ggplot2)
library(irr)

df <- read_csv("./annotations-with-truth.csv")

# Coarse grained (-1 and 1 become -2 and 2)
to_coarse <- function(x) {
  res <- 0
  if (x > 0){ res <- 2 }
  if (x < 0) { res <- -2 }
  res
}
to_coarse <- Vectorize(to_coarse)

df$coarse_bai <- to_coarse(df$delta_bai)
df$coarse_zining <- to_coarse(df$delta_zining)
df$coarse_upwork <- to_coarse(df$delta_upwork)
  
# Correlations. About 0.65 between me and Zining, 0.55 between upwork and either of us.
cor(df %>% select(delta_bai, delta_zining, delta_upwork), method="pearson")
cor(df %>% select(delta_bai, delta_zining, delta_upwork), method="spearman")
cor(df %>% select(coarse_bai, coarse_zining, coarse_upwork), method="pearson")
cor(df %>% select(coarse_bai, coarse_zining, coarse_upwork), method="spearman")

# Cohen's kappa. Moderate (0.4-0.5) for score, very low (0.1-0.2) for code.
kappa2(df %>% select(coarse_bai, coarse_zining))
kappa2(df %>% select(coarse_bai, coarse_upwork))
kappa2(df %>% select(coarse_zining, coarse_upwork))
kappa2(df %>% filter(coarse_bai != 0, coarse_zining != 0) %>% select(code_bai, code_zining))
kappa2(df %>% filter(coarse_bai != 0, coarse_upwork != 0) %>% select(code_bai, code_upwork))
kappa2(df %>% filter(coarse_zining != 0, coarse_upwork != 0) %>% select(code_zining, code_upwork))

# Percentage incorrect (opposite from ground truth). Upwork is worst with 14% wrong.
(df %>% filter(coarse_bai == -correct) %>% nrow) / nrow(df)
(df %>% filter(coarse_zining == -correct) %>% nrow) / nrow(df)
(df %>% filter(coarse_upwork == -correct) %>% nrow) / nrow(df)

# Distribution of annotations, code distribution is quite different.
ggplot(df, aes(x=delta_bai)) + geom_bar() + theme_bw()
ggplot(df, aes(x=delta_zining)) + geom_bar() + theme_bw()
ggplot(df, aes(x=delta_upwork)) + geom_bar() + theme_bw()
ggplot(df %>% filter(coarse_bai != 0), aes(x=code_bai)) + geom_bar() + theme_bw()
ggplot(df %>% filter(coarse_zining != 0), aes(x=code_zining)) + geom_bar() + theme_bw()
ggplot(df %>% filter(coarse_upwork != 0), aes(x=code_upwork)) + geom_bar() + theme_bw()

# Count rows at each level of score agreement.
# 50% of data completely agree (all 3 annotators)
# 35% of data mostly agree (2 annotators agree and 3rd differ by 1 coarse-level)
# 15% of data heavily disagree
df <- df %>% mutate(disagree =
                abs(coarse_bai - coarse_zining) +
                abs(coarse_bai - coarse_upwork) +
                abs(coarse_zining - coarse_upwork))
(df %>% filter(disagree == 0) %>% nrow) / nrow(df)
(df %>% filter(disagree <= 4) %>% nrow) / nrow(df)

# About 0.8 pearson correlation and 0.5-0.6 Cohen kappa if filtering out heavy disagree
df %>% filter(disagree <= 4) %>%
  select(coarse_bai, coarse_zining, coarse_upwork) %>%
  cor(method="pearson")
kappa2(df %>% filter(disagree <= 4) %>% select(coarse_bai, coarse_zining))
kappa2(df %>% filter(disagree <= 4) %>% select(coarse_bai, coarse_upwork))
kappa2(df %>% filter(disagree <= 4) %>% select(coarse_zining, coarse_upwork))
