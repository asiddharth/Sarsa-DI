#include "SarsaNAgent.h"
 #include <algorithm>
//add lambda as parameter to sarsaagent
SarsaNAgent::SarsaNAgent(int numFeatures, int numActions, double learningRate, double epsilon, double lambda, int nStep, FunctionApproximator *FA, char *loadWeightsFile, char *saveWeightsFile):PolicyAgent(numFeatures, numActions, learningRate, epsilon, FA, loadWeightsFile, saveWeightsFile){
  this->lambda = lambda;
  this->nStep = nStep;
  episodeNumber = 0;
  lastAction = -1;
  rewards = new double[nStep];
  actions = new int[nStep];
  states = new std::vector<double>[nStep];
  stepCount = 0;
  //have memory for lambda
}

void SarsaNAgent::update(double state[], int action, double reward, double discountFactor){

    if (stepCount < nStep) {
        rewards[stepCount] = reward;
        stepCount ++;
        return;
    } else {
        rewards[nStep-1] = reward;
    }
    stepCount++;
    
    FA->setState(lastState);

    double oldQ = FA->computeQ(lastAction);
    //std::cout << "::::::::::::::" <<lastAction; 
    //FA->updateTraces(lastAction);
    double delta = lastReward - oldQ;
    FA->setState(state);
    //Sarsa update
    double newQ = FA->computeQ(action);
    delta += discountFactor * newQ;

    FA->updateWeights(delta, learningRate);
    //Assume gamma, lambda are 0.
    //FA->decayTraces(discountFactor*lambda);//replace 0 with gamma*lambda

}

void SarsaNAgent::endEpisode(){
  int size = std::min(stepCount, nStep);
  lastReward = 0;
  for (int i = 0; i< size; i++) {
    lastReward += rewards[i];
  }
  
  episodeNumber++;
  //This will not happen usually, but is a safety.
  if(lastAction == -1){
    return;
  }
  else{
    for (int i = 0; i < size; i++) {
        for (int abc = 0; abc < getNumFeatures(); abc ++) {
            lastState[abc] = states[i][abc];
        }
        lastAction = actions[i];
        FA->setState(lastState);
        double oldQ = FA->computeQ(lastAction);
        //FA->updateTraces(lastAction);
        double delta = lastReward - oldQ;
        FA->updateWeights(delta, learningRate);
        //Assume lambda is 0. this comment looks wrong.
        //FA->decayTraces(0);//remains 0
        lastReward -= rewards[i];
    }
  }

  if(toSaveWeights && (episodeNumber + 1) % 5 == 0){
    saveWeights(saveWeightsFile);
    std::cout << "Saving weights to " << saveWeightsFile << std::endl;
  }

  lastAction = -1;

}

void SarsaNAgent::reset(){
  
  lastAction = -1;
}

int SarsaNAgent::selectAction(double state[]){
  int action;

  if(drand48() < epsilon){
    action = (int)(drand48() * getNumActions()) % getNumActions();
  }
  else{
    action = argmaxQ(state);
  }
  //std::cerr << "::::::::::::::1" <<lastAction;
  if (stepCount < nStep) {
    std::vector<double> tempstate;
    //std::cerr << "::::::::::::::6" <<lastAction;
    for(int i = 0; i < getNumFeatures(); i++){
      tempstate.push_back(state[i]);
    }
    states[stepCount] = tempstate;
    //std::cerr << "::::::::::::::7" <<lastAction;
    actions[stepCount] = action;
    if (stepCount == 0) {
        lastAction = action;
          //std::cerr << "::::::::::::::5" <<lastAction;
        for (int abc = 0; abc < getNumFeatures(); abc ++) {
            lastState[abc] = states[0][abc];
        }
    }
      //std::cerr << "::::::::::::::2" <<lastAction;
  } else {
        for (int abc = 0; abc < getNumFeatures(); abc ++) {
            lastState[abc] = states[0][abc];
        }
        lastAction = actions[0];
        lastReward = 0;
        for (int abc = 0; abc < nStep; abc ++) {
            lastReward += rewards[abc];
        }
        for (int abc = 0; abc < nStep-1; abc ++) {
            for(int i = 0; i < getNumFeatures(); i++){
                states[abc][i] = states[abc+1][i];
            }
            rewards[abc] = rewards[abc+1];
            actions[abc] = actions[abc+1];
        }
        for(int i = 0; i < getNumFeatures(); i++){
            states[nStep-1][i] = state[i];
        }
        actions[nStep-1] = action;
          //std::cerr << "::::::::::::::3" <<lastAction;
  }
  //std::cerr << "::::::::::::::4" <<lastAction;
  return action;
}

int SarsaNAgent::argmaxQ(double state[]){
  
  double Q[getNumActions()];

  FA->setState(state);

  for(int i = 0; i < getNumActions(); i++){
    Q[i] = FA->computeQ(i);
  }
  
  int bestAction = 0;
  double bestValue = Q[bestAction];
  int numTies = 0;

  double EPS=1.0e-4;

  for (int a = 1; a < getNumActions(); a++){

    double value = Q[a];
    if(fabs(value - bestValue) < EPS){
      numTies++;
      
      if(drand48() < (1.0 / (numTies + 1))){
	bestValue = value;
	bestAction = a;
      }
    }
    else if (value > bestValue){
      bestValue = value;
      bestAction = a;
      numTies = 0;
    }
  }
  
  return bestAction;
}

//Be careful. This resets FA->state.
double SarsaNAgent::computeQ(double state[], int action){

  FA->setState(state);
  double QValue = FA->computeQ(action);

  return QValue;
}

