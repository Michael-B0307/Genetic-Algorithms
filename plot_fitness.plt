# Set datafile format
set datafile separator ","

# Set labels and title for the plot
set xlabel "Generation"
set ylabel "Average Fitness"
set title "Average Fitness over Generations"

# Plotting data from the different fitness csv files.
# Column 1: Generation number
# Column 2: Average fitness value

set key bottom right
# plot "fitness_history.csv" using 1:2 with lines title "One-Max Problem Fitness"
# plot "fitness_history_target.csv" using 1:2 with lines title "Evolving to Target String Fitness"
# plot "fitness_history_deceptive.csv" using 1:2 with lines title "Deceptive Landscape Fitness"
