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

sd(game_lengths$T)
mean(game_lengths$T)
min(game_lengths$T)
max(game_lengths$T)

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

# training stats
training_stat_dfs <- list()
for (model in list.files("../checkpoints/nba")) {
        fp <- paste("../checkpoints/nba/", model, "/_training_stats.jsonl",
                sep = ""
        )
        training_stat_dfs[[model]] <-
                jsonlite::stream_in(file(fp)) %>%
                mutate(model = model)
        print(df)
}

training_stats <- bind_rows(training_stat_dfs) %>%
        select(model, everything()) %>%
        rename(Epoch = epoch) %>%
        separate(model, into = c("RNN type", "Model size"), sep = "-") %>%
        mutate(
                `RNN type` = factor(`RNN type`),
                `Model size` = factor(`Model size`,
                        levels = c("xs", "sm", "md", "lg", "xl")
                ),
        ) %>%
        arrange(`RNN type`, `Model size`, Epoch)

plots$training_curves <-
        training_stats %>%
        select(-ends_with("loss")) %>%
        pivot_longer(ends_with("accuracy"),
                names_transform = \(c) str_split_i(c, "_", 1),
                names_to = "Split",
                values_to = "Accuracy"
        ) %>%
        ggplot(aes(Epoch, Accuracy, color = Split)) +
        geom_line() +
        facet_grid(`Model size` ~ `RNN type`)

# best validation accuracy from each model
best_epochs <- training_stats %>%
        group_by(`RNN type`, `Model size`) %>%
        slice_max(order_by = test_accuracy)

# shapley plots
plots$shapley <- read_csv(".shapley-vals.txt", col_names = "Shapley Value") %>%
        mutate(Index = 1:n() - 1) %>%
        ggplot(aes(Index, `Shapley Value`)) +
        geom_col()

# save plots
big_plots <- c("training_curves")
for (plot_name in names(plots)) {
        filepath <- paste("../paper/figures/", plot_name, ".tikz", sep = "")
        print(paste("Saving", filepath, "..."))
        ggsave(
                filepath, plots[[plot_name]],
                device = tikzDevice::tikz,
                width = ifelse(plot_name %in% big_plots, 5.5, 5),
                height = ifelse(plot_name %in% big_plots, 8, 3),
        )
}
