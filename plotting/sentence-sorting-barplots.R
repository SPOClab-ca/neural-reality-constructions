library(tidyverse)
library(ggplot2)


# Left half of fig:human-lm-sorting-deviation
human_results_df = data.frame(
  Level=c("Beginner", "Intermediate", "Advanced", "Native"),
  CDev=c(7.239, 5.290, 4.909, 4.45),
  VDev=c(5.804, 7.226, 8.242, 7.65)
) %>%
  mutate(Level=factor(Level, levels=Level))

human_results_df %>%
  gather(type, Value, CDev, VDev) %>%
  ggplot(aes(x=Level, y=Value, fill=type)) +
    geom_bar(position="dodge", stat="identity") +
    scale_y_continuous(breaks=c(0, 2, 4, 6, 8, 10, 12), limits=c(0, 12)) +
    scale_fill_brewer(palette="Dark2") +
    theme_minimal() +
    theme(text = element_text(size=14))

ggsave("sentence-sorting-human.pdf", width=6, height=4.5)


# Right half of fig:human-lm-sorting-deviation
lm_results_df = data.frame(
  Level=c("1M", "10M", "100M", "1B", "30B"),
  CDev=c(9.690, 7.490, 4.860, 4.930, 3.880),
  VDev=c(6.300, 9.020, 9.540, 9.640, 10.910),
  CI_CDev=c(0.072, 0.096, 0.113, 0.126, 0.132),
  CI_VDev=c(0.074, 0.044, 0.033, 0.033, 0.018)
) %>%
  mutate(Level=factor(Level, levels=Level))

lm_results_df %>%
  gather(type, Value, CDev, VDev) %>%
  mutate(ci=if_else(type == "CDev", CI_CDev, CI_VDev)) %>%
  ggplot(aes(x=Level, y=Value, fill=type)) +
    geom_bar(position="dodge", stat="identity") +
    scale_y_continuous(breaks=c(0, 2, 4, 6, 8, 10, 12), limits=c(0, 12)) +
    xlab("Pretraining Data Size") +
    geom_errorbar(aes(ymin=Value-ci, ymax=Value+ci),
                  position=position_dodge(0.9), width=0.3, size=1) +
    scale_fill_brewer(palette="Dark2") +
    theme_minimal() +
    theme(text = element_text(size=14))

ggsave("sentence-sorting-lm.pdf", width=6, height=4.5)


# Left half of fig:multilang-sorting
multilang_human_df = data.frame(
  Level=c("German", "Italian B1", "Italian B2", "Spanish"),
  CDev=c(10.175, 5.3, 3.5, 6.194),
  VDev=c(2.5, 7.4, 8.5, 5.806)
) %>%
  mutate(Level=factor(Level, levels=Level))

multilang_human_df %>%
  gather(type, Value, CDev, VDev) %>%
  ggplot(aes(x=Level, y=Value, fill=type)) +
    geom_bar(position="dodge", stat="identity") +
    scale_y_continuous(breaks=c(0, 2, 4, 6, 8, 10, 12), limits=c(0, 12)) +
    scale_fill_brewer(palette="Dark2") +
    theme_minimal() +
    theme(text = element_text(size=14))

ggsave("multilang-sorting-human.pdf", width=6, height=4.5)


# Right half of fig:multilang-sorting
multilang_lm_df = data.frame(
  Level=c("German (mBERT)", "German (mono)", "Italian (mBERT)", "Italian (mono)",
          "Spanish (mBERT)", "Spanish (mono)"),
  CDev=c(6, 7, 8, 7, 6, 6),
  VDev=c(10, 10, 9, 8, 10, 11)
) %>%
  mutate(Level=factor(Level, levels=Level))

multilang_lm_df %>%
  gather(type, Value, CDev, VDev) %>%
  ggplot(aes(x=Level, y=Value, fill=type)) +
    geom_bar(position="dodge", stat="identity") +
    scale_y_continuous(breaks=c(0, 2, 4, 6, 8, 10, 12), limits=c(0, 12)) +
    scale_fill_brewer(palette="Dark2") +
    theme_minimal() +
    theme(text = element_text(size=14))

ggsave("multilang-sorting-lm.pdf", width=6, height=4.5)
