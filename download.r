# R: install packages (run once)
install.packages(c("dplyr","lubridate","readr"))
# install nflreadr from CRAN or github
install.packages("nflreadr") # CRAN currently provides nflreadr
# OR if y7-ou want nflfastR codebase:
# install.packages("devtools")
# devtools::install_github("mrcaseb/nflfastR")

library(nflreadr)
library(dplyr)
library(lubridate)
library(readr)

# 1.1 — choose seasons to download
seasons <- 2024:2025

# 1.2 — load play-by-play for those seasons
pbp_all <- load_pbp(seasons)   # loads a dataframe with pbp rows

# 1.3 — quick sanity inspect
glimpse(pbp_all)
# check unique seasons, number of rows
table(pbp_all$season)

# 1.4 — save to compressed CSV for Python use
readr::write_csv(pbp_all, "pbp_2023_2024.csv")
