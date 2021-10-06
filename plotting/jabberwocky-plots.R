library(tidyverse)
library(ggplot2)

high_freq_df = data.frame(
  cxn=c("ditransitive", "ditransitive", "ditransitive", "ditransitive",
        "resultative", "resultative", "resultative", "resultative",
        "caused-motion", "caused-motion", "caused-motion", "caused-motion",
        "removal", "removal", "removal", "removal"),
  verb=c("gave", "made", "put", "took",
         "gave", "made", "put", "took",
         "gave", "made", "put", "took",
         "gave", "made", "put", "took"),
  values=c(11.899, 12.295, 12.567, 12.328,
           11.924, 11.701, 11.868, 11.864,
           11.691, 11.593, 11.395, 11.599,
           11.740, 11.954, 11.936, 11.517)
) %>%
  mutate(cxn=factor(cxn, levels=rev(c(
    "ditransitive", "resultative", "caused-motion", "removal")))) %>%
  mutate(verb=factor(verb, levels=c(
    "gave", "made", "put", "took")))

low_freq_df = data.frame(
  cxn=c("ditransitive", "ditransitive", "ditransitive", "ditransitive",
        "resultative", "resultative", "resultative", "resultative",
        "caused-motion", "caused-motion", "caused-motion", "caused-motion",
        "removal", "removal", "removal", "removal"),
  verb=c("handed", "turned", "placed", "removed",
         "handed", "turned", "placed", "removed",
         "handed", "turned", "placed", "removed",
         "handed", "turned", "placed", "removed"),
  values=c(12.008, 12.939, 13.141, 13.677,
           12.230, 12.466, 12.562, 13.095,
           11.791, 12.142, 11.906, 12.742,
           11.860, 12.420, 12.651, 12.246)
) %>%
  mutate(cxn=factor(cxn, levels=rev(c(
    "ditransitive", "resultative", "caused-motion", "removal")))) %>%
  mutate(verb=factor(verb, levels=c(
    "handed", "turned", "placed", "removed")))

# fig:jg-grid-high-freq
ggplot(high_freq_df, aes(verb, cxn, fill=values)) + 
  geom_tile() +
  xlab("") +
  ylab("") +
  scale_x_discrete(position = "top") +
  scale_fill_distiller(palette = "RdYlGn",
    limits = c(min(high_freq_df$values) - 0.1, max(high_freq_df$values) + 0.1)) +
  geom_text(aes(label = sprintf("%0.3f", values)), color="#303030") +
  theme_minimal() +
  theme(legend.position = "none") +
  theme(text = element_text(size=15))

ggsave("jg-grid-high-freq.pdf", width=4.5, height=3.5)

# fig: jg-grid-low-freq
ggplot(low_freq_df, aes(verb, cxn, fill=values)) + 
  geom_tile() +
  xlab("") +
  ylab("") +
  scale_x_discrete(position = "top") +
  scale_fill_distiller(palette = "RdYlGn",
    limits = c(min(low_freq_df$values) - 0.1, max(low_freq_df$values) + 0.1)) +
  geom_text(aes(label = sprintf("%0.3f", values)), color="#303030") +
  theme_minimal() +
  theme(legend.position = "none") +
  theme(text = element_text(size=15))

ggsave("jg-grid-low-freq.pdf", width=4.5, height=3.5)


# fig:jg-barplot-high-freq
high_freq_bar_df = data.frame(
  Condition=c("Congruent", "Incongruent"),
  Distance=c(11.628, 11.947),
  CI_Distance=c(0.036, 0.023)
) %>%
  mutate(Condition=factor(Condition, levels=Condition))

high_freq_bar_df %>%
  ggplot(aes(x=Condition, y=Distance)) +
    geom_bar(stat="identity") +
    xlab("") +
    ylab("Euclidean distance") +
    coord_cartesian(ylim = c(11, 13)) + 
    geom_errorbar(aes(ymin=Distance-CI_Distance, ymax=Distance+CI_Distance),
                  position=position_dodge(0.9), width=0.3, size=1) +
    theme_minimal() +
    theme(text = element_text(size=14))

ggsave("jg-barplot-high-freq.pdf", width=3, height=3)


# fig:jg-barplot-low-freq
low_freq_bar_df = data.frame(
  Condition=c("Congruent", "Incongruent"),
  Distance=c(12.157, 12.604),
  CI_Distance=c(0.038, 0.026)
) %>%
  mutate(Condition=factor(Condition, levels=Condition))

low_freq_bar_df %>%
  ggplot(aes(x=Condition, y=Distance)) +
    geom_bar(stat="identity") +
    xlab("") +
    ylab("Euclidean distance") +
    coord_cartesian(ylim = c(11, 13)) + 
    geom_errorbar(aes(ymin=Distance-CI_Distance, ymax=Distance+CI_Distance),
                  position=position_dodge(0.9), width=0.3, size=1) +
    theme_minimal() +
    theme(text = element_text(size=14))

ggsave("jg-barplot-low-freq.pdf", width=3, height=3)
