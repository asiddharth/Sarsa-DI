#include "tdlambdanstepagent.h"
#include <math.h>
TDLambdaNStepAgent::TDLambdaNStepAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const double &initWeight, const double &lambda, const double &alphaStart, const double &epsilonStart, const int &randomSeed, const int &persStepSizes) : Agent(numFeatures, numActions, randomSeed){

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

  this->lambda = lambda;

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
  persStepSize = persStepSizes;
  stepCount = 0;
}

TDLambdaNStepAgent::~TDLambdaNStepAgent(){

}


void TDLambdaNStepAgent::resetEligibility(){

  eligibility.resize(numActions);
  for(int a = 0; a < numActions; a++){
    eligibility[a].resize(numFeatures);

    for(int f = 0; f < numFeatures; f++){
      eligibility[a][f] = 0;
    }
  }
}


double TDLambdaNStepAgent::computeQ(const vector<double> &state, const int &action){

  double v = 0;

  for(int f = 0; f < numFeatures; f++){
    v += weights[action][f] * state[f];
  }

  return v;
}

int TDLambdaNStepAgent::argMaxQ(const vector<double> &state){

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

int TDLambdaNStepAgent::getnumActions() {
	return numActions;
}

int TDLambdaNStepAgent::takeAction(const vector<double> &state){

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

  if(gsl_rng_uniform(ran) < epsilon){
    action = (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }
  else{
    action = argMaxQ(state);
  }

  if (stepCount % persStepSize != 0) {
	  return lastAction;
  }
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
    //std :: cout << alpha << " " << delta << " " <<denominator << "\n";
    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
	weights[a][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);

    //std :: cout << weights[a][f] << " ";
      }
      //std :: cout << "\n";
    }
  }

  for(int i = 0; i < numFeatures; i++){
    lastState[i] = state[i];
  }

  lastAction = action;
  return action;
}

int TDLambdaNStepAgent::takeBestAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }


  return (argMaxQ(state));
}


void TDLambdaNStepAgent::update(const double &reward, const vector<double> &state, const bool &terminal){


  if(diverged){
    return;
  }

  if (stepCount % persStepSize != 0) {
	  double gamma = 1;
	  lastReward += pow(gamma, stepCount) * reward;
	  stepCount ++;
	  if (stepCount %persStepSize == 0) {
		  stepCount = 0;
	  }
   } else if  (stepCount % persStepSize == 0){
	   lastReward = reward;
	   stepCount ++;
   }

  if(terminal){


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

    resetEligibility();

    lastAction = -1;

    numEpisodesDone++;

    alpha = alphaK1 / (alphaK2 + numEpisodesDone - 1);
    epsilon = epsilonK1 / (epsilonK2 + numEpisodesDone - 1);
    stepCount = 0;
  }

}

vector<double> TDLambdaNStepAgent::getWeights() {

  vector<double> w;
  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){
      w.push_back(weights[a][f]);
    }
  }
  return w;
}

vector<vector <double> >  TDLambdaNStepAgent::getWeightsFull() {
	return weights;
}


