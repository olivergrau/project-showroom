#!/bin/bash

# Array of time limits to iterate over
time_limits=(150 300 450 600)

# Number of rounds (-r) and parallel processes (-p)
rounds=50
processes=16

# Loop over the time limits and run the command
for time_limit in "${time_limits[@]}"; do
  echo "Running match with time limit: $time_limit ms"
  python run_match.py -f -r $rounds -p $processes -t $time_limit
  
  # Optionally, add a pause between runs (uncomment the next line if needed)
  # sleep 1
done
