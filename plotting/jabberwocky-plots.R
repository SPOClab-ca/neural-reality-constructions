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
)

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
)

# WIP
ggplot(high_freq_df, aes(verb, cxn, fill=values)) + 
  geom_tile()