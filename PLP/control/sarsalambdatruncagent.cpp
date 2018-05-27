#include "sarsalambdatruncagent.h"

SarsaLambdaTruncAgent::SarsaLambdaTruncAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const double &initWeight, const double &lambda, const double &alphaStart, const double &epsilonStart, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){
  
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
  lastStateActionValue = 0;

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

}

SarsaLambdaTruncAgent::~SarsaLambdaTruncAgent(){
  
}


void SarsaLambdaTruncAgent::resetEligibility(){

  eligibility.resize(numActions);
  for(int a = 0; a < numActions; a++){
    eligibility[a].resize(numFeatures);

    for(int f = 0; f < numFeatures; f++){
      eligibility[a][f] = 0;
    }
  }
}


double SarsaLambdaTruncAgent::computeQ(const vector<double> &state, const int &action){

  double v = 0;
  
  for(int f = 0; f < numFeatures; f++){
    v += weights[action][f] * state[f];
  }

  return v;
}

int SarsaLambdaTruncAgent::argMaxQ(const vector<double> &state){

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


int SarsaLambdaTruncAgent::takeAction(const vector<double> &state){

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
  
  if(lastAction != -1){
    double currentStateActionValue = computeQ(state, action);
    double delta = lastReward + currentStateActionValue - lastStateActionValue;
    double eUpdate = 0;
    for(int f = 0; f < numFeatures; f++){
        eUpdate += eligibility[lastAction][f] * lastState[f];
    }
    double denominator = 0;
    for(int f = 0; f < numFeatures; f++){
      denominator += lastState[f];
    }

    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
    	  eligibility[a][f] *= lambda;
    	  if (a == lastAction) {
    		  eligibility[a][f]	+= (alpha*(1-lambda*eUpdate)*lastState[f]) * (1.0 / denominator);
    	  }
      }
    }
    /*
    for(int a = 0; a < numActions; a++){
          for(int f = 0; f < numFeatures; f++){
    	eligibility[a][f] *= lambda;
     }



    for(int f = 0; f < numFeatures; f++){
      eligibility[lastAction][f] += lastState[f]; // Must be zero or one.
      if(eligibility[lastAction][f] > 1.0){
      	eligibility[lastAction][f] = 1.0;
      }
    }*/

    double prevValue = computeQ(lastState, lastAction);
    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
    	  double beforeweight = weights[a][f];
    	  weights[a][f] += ((delta * eligibility[a][f]))* (1.0 / denominator);
    	  if (a == lastAction) {
    		  weights[a][f] += (alpha*(lastStateActionValue-prevValue)*lastState[f])* (1.0 / denominator);
    		 // if (lastState[f] == 1)
    			  //std :: cout << (alpha*(lastStateActionValue-prevValue)*lastState[f]) * (1.0 / denominator) << " " << alpha << " " <<lastStateActionValue << " " <<prevValue << " " << lastState[f] << " " <<weights[a][f]<< " " << beforeweight <<" \n";
    	 }
    	  //std :: cout << weights[a][f] << " " << denominator << "\n";
      }
    }
    lastStateActionValue = currentStateActionValue;
  } else {
	  lastStateActionValue = computeQ(state, action);
  }

  for(int i = 0; i < numFeatures; i++){
    lastState[i] = state[i];
  }

  lastAction = action;

  return action;
}

int SarsaLambdaTruncAgent::takeBestAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  
  return (argMaxQ(state));
}


void SarsaLambdaTruncAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  
  if(diverged){
    return;
  }

  lastReward = reward;

  if(terminal){
    double delta = lastReward - lastStateActionValue;
    double eUpdate = 0;
    for(int f = 0; f < numFeatures; f++){
        eUpdate += eligibility[lastAction][f] * lastState[f];
    }

    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
     	  eligibility[a][f] *= lambda;
       	  if (a == lastAction) {
       		  eligibility[a][f]	+= (alpha*(1-lambda*eUpdate)*lastState[f]);
       	  }
      }
    }

    /*for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
    	  eligibility[a][f] *= lambda;
      }
    }
    for(int f = 0; f < numFeatures; f++){

      eligibility[lastAction][f] += lastState[f]; // Must be zero or one.
      if(eligibility[lastAction][f] > 1.0){
    	  eligibility[lastAction][f] = 1.0;
      }

    }*/

    double denominator = 0;
    for(int f = 0; f < numFeatures; f++){
      denominator += lastState[f];
    }

    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
       	  weights[a][f] += ((delta * eligibility[a][f]))* (1.0 / denominator);
       	  if (a == lastAction) {
       		  weights[a][f] += (alpha*(lastStateActionValue-computeQ(lastState, lastAction))*lastState[f])* (1.0 / denominator);
       	  }
      }
    }

    /*for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
		weights[a][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);
      }
    }*/

    resetEligibility();

    lastAction = -1;
    lastStateActionValue = 0;
    numEpisodesDone++;

    alpha = alphaK1 / (alphaK2 + numEpisodesDone - 1);
    epsilon = epsilonK1 / (epsilonK2 + numEpisodesDone - 1);
  }

}

vector<double> SarsaLambdaTruncAgent::getWeights(){

  vector<double> w;
  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){
      w.push_back(weights[a][f]);
    }
  }

  return w;
}

