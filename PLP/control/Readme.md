# Running Agents on the PLP control task

Run `make` from this folder to create the executable for the PLP agents. Run the following commands for different variants of Sarsa agents.

## DI-Sarsa(_&lambda;_) PLP Agent
`./experiment --s <s> --p <p> --chi <chi> --w <w> --sigma <sigma> --method tdnsteplambda_<lambda>_0.99_0.1_0_<di> --outFile <outfile> --totalEpisodes <num_episodes>  --displayInterval <display_interval> --randomSeed <random_seed>`

## True Online Sarsa(_&lambda;_) PLP Agent

`./experiment --s <s> --p <p> --chi <chi> --w <w> --sigma <sigma> --method truesarsa_<lambda>_0.1_0.1_0 --outFile <outfile> --totalEpisodes <num_episodes>  --displayInterval <display_interval> --randomSeed <random_seed>`

## Sarsa-n PLP Agent PLP Agent

`./experiment --s <s> --p <p> --chi <chi> --w <w> --sigma <sigma> --method tdnstep_0_0.99_0.1_0_<n> --outFile <outfile> --totalEpisodes <num_episodes>  --displayInterval <display_interval> --randomSeed <random_seed>`

