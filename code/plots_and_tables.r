library(tidyverse)
theme_set(theme_bw())

df <- bind_rows(
        read_csv("../data/nba-raw/NBA_PBP_2015-16.csv"),
        read_csv("../data/nba-raw/NBA_PBP_2016-17.csv"),
        read_csv("../data/nba-raw/NBA_PBP_2017-18.csv"),
        read_csv("../data/nba-raw/NBA_PBP_2018-19.csv"),
        read_csv("../data/nba-raw/NBA_PBP_2019-20.csv"),
        read_csv("../data/nba-raw/NBA_PBP_2020-21.csv"),
) %>% select(-`...41`) # drop weird column

plots <- list()

game_lengths <- df %>%
        group_by(URL) %>%
        summarise(T = n())

plots$game_length_hist <-
        game_lengths %>%
        ggplot(aes(T)) +
        geom_histogram() +
        labs(x = "Events per Game", y = "Count") +
        geom_vline(xintercept = mean(game_lengths$T), color = "red", lty = "dashed")

n_by_type <- df %>%
        summarize(
                `Total` = n(),
                `Shots` = sum(!is.na(ShotType)),
                `Free Throws` = sum(!is.na(FreeThrowOutcome)),
                `Rebounds` = sum(!is.na(ReboundType)),
                `Fouls` = sum(!is.na(FoulType)),
                `Violations` = sum(!is.na(ViolationType)),
                `Turnovers` = sum(!is.na(TurnoverType)),
                `Jumpballs` = sum(!is.na(FoulType)),
        ) %>%
        pivot_longer(everything(), names_to = "Event Type", values_to = "Count") %>%
        arrange(-Count) %>%
        mutate(`Event Type` = as_factor(`Event Type`))

n_by_type %>%
        dplyr::filter(`Event Type` != "Total") %>%
        ggplot(aes(x = `Event Type`, y = Count)) +
        geom_col()

plots$events_by_time <-
        df %>%
        mutate(Quarter = as_factor(Quarter)) %>%
        dplyr::filter(Quarter %in% c(1, 2, 3, 4)) %>%
        ggplot(aes(x = SecLeft, fill = Quarter, color = Quarter)) +
        stat_density(alpha = .2, position = "identity") +
        labs(x = "Seconds Left in Quarter", y = "Event Density")

# save plots
for (plot_name in names(plots)) {
        filepath <- paste("../paper/figures/", plot_name, ".tikz", sep = "")
        print(paste("Saving", filepath, "..."))
        ggsave(
                filepath, plots[[plot_name]],
                device = tikzDevice::tikz,
                width = 6,
                height = 3.5,
        )
}
