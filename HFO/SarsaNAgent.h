#ifndef SARSA_AGENT
#define SARSA_AGENT

#include "PolicyAgent.h"
#include "FuncApprox.h"
#include <vector>

class SarsaNAgent:public PolicyAgent{

 private:

  int episodeNumber;
  double lastState[MAX_STATE_VARS];
  int lastAction;
  double lastReward;
  double lambda;
  

 public:
  int nStep;
  int stepCount;
  std::vector <double>* states;
  int* actions;
  double* rewards;
  SarsaNAgent(int numFeatures, int numActions, double learningRate, double epsilon, double lambda, int nStep, FunctionApproximator *FA, char *loadWeightsFile, char *saveWeightsFile);

  int  argmaxQ(double state[]);
  double computeQ(double state[], int action);

  int selectAction(double state[]);

  void update(double state[], int action, double reward, double discountFactor);
  void endEpisode();
  void reset();

};

#endif

