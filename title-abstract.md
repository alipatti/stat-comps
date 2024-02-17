# Sports as Sequences: Win Prediction with Deep Learning

On Saturday, February 10, Manchester City played Everton in the English Premier
League. For the first two thirds of the game, things looked even on paper: the
score was tied at 0-0, and both teams had just one shot on target. Yet any fan
watching the game would say that Manchester City was dominating. This heuristic
evaluation eventually was proven correct: City broke the deadlock and scored
twice in the last 20 minutes to win 2-0.

In this talk, we develop a win-prediction model that emulates this "eye test"
and incorporates game dynamics not appearing in summary statistics. To do this,
we model basketball and soccer games as sequences of events (for example passes,
shots, turnovers, and fouls) and use these sequences to predict winners using a
form of deep learning called a recurrent neural network.
