# Making the agent executables
1. Clone `https://github.com/asiddharth/HFO/commits/master`.
2. Follow the Readme to install Half Field Offense.
3. Copy over `policyMakeFile` to `example/sarsa_libraries/policy` and `funcapproxMakefile` to `example/sarsa_libraries/funcapprox`.
4. Copy over `SarsaNAgent.cpp` and `SarsaNAgent.h` to `example/sarsa_libraries/policy`
5. Rename both `policyMakeFile` and `funcapproxMakefile` as `MakeFile`.
6. Copy over `MakeFile`,`high_level_di-sarsa_defense_agent.cpp` and `high_level_sarsa_n_defense_agent.cpp` to `example/sarsa_defense`
7. In the MakeFile :
	-Change `SRC = agent source` and `TARGET = agent target` to create an executable for the required agent.
   The included MakeFile will create an executable named `high_level_di-sarsa_defense_agent` from the DI-Sarsa agent source `high_level_di-sarsa_defense_agent.cpp`
8. Run make from `example/sarsa_defense` to create the agent executable.


# Running the agents

## DI-Sarsa(_&lambda;_) HFO Agent 
## Start the HFO Server
From the root of the github repository, execute the following command to start the HFO server. 

>`./bin/HFO --offense-npcs 2 --defense-agents 1 --defense-npcs 1 --trials <num_episodes> --headless --no-logging --port <server_port>`

## Run the agent
Change directory to `example/sarsa_defense` and execute the following command to start the DI-Sarsa(_&lambda;_) HFO Agent.

>`./high_level_di-sarsa_defense_agent --numAgents 1 --basePort <server_port> --numOpponents 2 --step <di> --weightId <weightId> --numEpisodes <num_episodes>`

weightId is used to differentiate between different HFO training runs of the same agent.

## Sarsa-n HFO Agent

In the MakeFile in `example/sarsa_defense` :
	-Change `SRC = high_level_sarsa_n_defense_agent.cpp` and `TARGET = high_level_sarsa_n_defense_agent` and run make to create the executable.

## Start the HFO Server
From the root of the github repository, execute the following command to start the HFO server. 

>`./bin/HFO --offense-npcs 2 --defense-agents 1 --defense-npcs 1 --trials <num_episodes> --headless --no-logging --port <server_port>`

## Run the agent
Change directory to `example/sarsa_defense` and execute the following command to start the DI-Sarsa(_&lambda;_) HFO Agent.

>`./high_level_sarsa_n_defense_agent --numAgents 1 --basePort <server_port> --numOpponents 2 --step <n> --weightId <weightId> --numEpisodes <num_episodes>`

weightId is used to differentiate between different HFO training runs of the same agent.


The executables of both agents are `high_level_di-sarsa_defense_agent` and `high_level_sarsa_n_defense_agent` included for convenience, but might need to be re-made.




