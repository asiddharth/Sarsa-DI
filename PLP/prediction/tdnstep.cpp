#include "tdnstep.h"
#include <math.h>
TDNStep::TDNStep(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const double &initWeight, const double &lambda, const double &alphaStart, const double &epsilonStart, const int &randomSeed, const int &persStepSizes) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;

  // Set all weights to 0 initially.
  // Set all weights to initWeight initially.
  for(int a = 0; a < numActions; a++){

    vector<double> w;
    for(int f = 0; f < numFeatures; f++){
      //      w.push_back(0);
      w.push_back(initWeight);
    }

    weights.push_back(w);
  }

  for(int f = 0; f < numFeatures; f++){
    lastState.push_back(0);
  }
  lastAction = -1;
  lastReward = 0;

  numEpisodesDone = 0;

  this->lambda = 0;

  // Clear history trajectory.
  resetEligibility();

  double alphaEnd = 0.01;
  alpha = alphaStart;
  alphaK2 = (alphaEnd * (totalEpisodes - 1)) / (alphaStart - alphaEnd);
  alphaK1 = alphaStart * alphaK2;

  double epsilonEnd = 0.01;
  epsilon = epsilonStart;
  epsilonK2 = (epsilonEnd * (totalEpisodes - 1)) / (epsilonStart - epsilonEnd);
  epsilonK1 = epsilonStart * epsilonK2;

  diverged = false;
  nStep = persStepSizes;
  stepCount = 0;
  rewards = new double[nStep];
  actions = new int[nStep];
  states = new vector<double>[nStep];
}

TDNStep::~TDNStep(){

}


void TDNStep::resetEligibility(){

  eligibility.resize(numActions);
  for(int a = 0; a < numActions; a++){
    eligibility[a].resize(numFeatures);

    for(int f = 0; f < numFeatures; f++){
      eligibility[a][f] = 0;
    }
  }
}


double TDNStep::computeQ(const vector<double> &state, const int &action){

  double v = 0;

  for(int f = 0; f < numFeatures; f++){
    v += weights[action][f] * state[f];
  }

  return v;
}

int TDNStep::argMaxQ(const vector<double> &state){

  int bestAction = 0;
  double bestVal = computeQ(state, 0);

  int numTies = 0;

  for(int a = 1; a < numActions; a++){

    double val = computeQ(state, a);
    if(fabs(val - bestVal) < EPS){

      numTies++;
      if(gsl_rng_uniform(ran) < (1.0 / (1.0 + numTies))){
	bestVal = val;
	bestAction = a;
      }
    }
    else if(val > bestVal){
      bestVal = val;
      bestAction = a;
      numTies = 0;
    }
  }

  return bestAction;
}

int TDNStep::getnumActions() {
	return numActions;
}

int TDNStep::takeAction(const vector<double> &state1){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){

      if(isinf(weights[a][f]) || (weights[a][f] != weights[a][f])){
	diverged = true;
	cout << "Diverged\n";
      }
    }
  }

  int action;
/*
  if(gsl_rng_uniform(ran) < epsilon){
    action = (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }
  else{
    action = argMaxQ(state);
  }
*/
  vector <double> state;
  for (int indexx = 0; indexx < numFeatures; indexx++) {
	  state.push_back(state1[indexx]);
  }
  action = (int)state1[state1.size()-1];
  if (stepCount < nStep) {
	  /*cout << "STATE" << " ";
	  for (int i = 0; i< numFeatures; i++) {
		  cout << state[i] << " ";
	  }
	  cout << "\n LAST STATE" << " ";
	  for (int i = 0; i< numFeatures; i++) {
	  	  cout << lastState[i] << " ";
	  }
	  cout << "\n ACTION " << action << " LAST ACTION " << lastAction << "LAST REWARD " << lastReward<< "\n";*/
	  states[stepCount] = state;
	  actions[stepCount] = action;
	  if (stepCount == 0) {
		  lastAction = action;
		  for (int abc = 0; abc < numFeatures; abc ++) {
			  lastState[abc] = states[0][abc];
		  }
	  }
	  return action;
  }

  for (int abc = 0; abc < numFeatures; abc ++) {
  	  		  lastState[abc] = states[0][abc];
  }
  lastAction = actions[0];
  lastReward = 0;
  for (int abc = 0; abc < nStep; abc ++) {
	  lastReward += rewards[abc];
  }
  for (int abc = 0; abc < nStep-1; abc ++) {
	  states[abc] = states[abc+1];
	  rewards[abc] = rewards[abc+1];
	  actions[abc] = actions[abc+1];
  }
  states[nStep-1] = state;
  actions[nStep-1] = action;
  /*cout << "STATE" << " ";
  for (int i = 0; i< numFeatures; i++) {
	  cout << state[i] << " ";
  }
  cout << "\n LAST STATE" << " ";
  for (int i = 0; i< numFeatures; i++) {
  	  cout << lastState[i] << " ";
  }
  cout << "\n ACTION " << action << " LAST ACTION " << lastAction << "LAST REWARD " << lastReward << "\n";*/
  if(lastAction != -1){

    double delta = lastReward + computeQ(state, action) - computeQ(lastState, lastAction);

    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
	eligibility[a][f] *= lambda;
      }
    }
    for(int f = 0; f < numFeatures; f++){
    	eligibility[lastAction][f] += lastState[f]; // Must be zero or one.
      if(eligibility[lastAction][f] > 1.0){
      	eligibility[lastAction][f] = 1.0;
      }

    }

    double denominator = 0;
    for(int f = 0; f < numFeatures; f++){
      denominator += lastState[f];
    }
    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
    	  weights[a][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);
      }
    }
  }

  return action;
}

int TDNStep::takeBestAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }


  return (int)state[state.size()-1];
}


void TDNStep::update(const double &reward, const vector<double> &state, const bool &terminal){


  if(diverged){
    return;
  }
  if (stepCount < nStep)
	  rewards[stepCount] = reward;
  else
	  rewards[nStep-1] = reward;
  stepCount++;
  bool finished = false;
  int size = min(stepCount, nStep);

  /*cout << "TERMINAL";
  cout << "STATE" << " ";
  for (int i = 0; i< numFeatures; i++) {
	  cout << state[i] << " ";
  }
  cout << "\n LAST STATE" << " ";
  for (int i = 0; i< numFeatures; i++) {
  	  cout << lastState[i] << " ";
  }
  cout <<  "\n LAST ACTION " << lastAction <<  "LAST REWARD " << lastReward<<"\n";*/
  lastReward = 0;
  for (int i = 0; i< size; i++) {
	  lastReward += rewards[i];
  }
  if(terminal){
	for (int i = 0; i < size; i++) {
		for (int abc = 0; abc < numFeatures; abc ++) {
			lastState[abc] = states[i][abc];
		}
		lastAction = actions[i];
		double delta = lastReward - computeQ(lastState, lastAction);
		for(int a = 0; a < numActions; a++){
			for(int f = 0; f < numFeatures; f++){
				eligibility[a][f] *= lambda;
			}
		}
		for(int f = 0; f < numFeatures; f++){
			eligibility[lastAction][f] += lastState[f]; // Must be zero or one.
			if(eligibility[lastAction][f] > 1.0){
				eligibility[lastAction][f] = 1.0;
			}
		}

		double denominator = 0;
		for(int f = 0; f < numFeatures; f++){
			denominator += lastState[f];
		}

		for(int a = 0; a < numActions; a++){
			for(int f = 0; f < numFeatures; f++){
				weights[a][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);
			}
		}
		lastReward -= rewards[i];
	}

    resetEligibility();

    lastAction = -1;

    numEpisodesDone++;

    alpha = alphaK1 / (alphaK2 + numEpisodesDone - 1);
    epsilon = epsilonK1 / (epsilonK2 + numEpisodesDone - 1);
    stepCount = 0;
    delete[] states;
    delete[] actions;
    delete[] rewards;
    states = new vector<double>[nStep];
    actions = new int[nStep];
    rewards = new double[nStep];
  }

}

vector<double> TDNStep::getWeights() {

  vector<double> w;
  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){
      w.push_back(weights[a][f]);
    }
  }
  return w;
}

vector<vector <double> >  TDNStep::getWeightsFull() {
	return weights;
}


