#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <math.h>
#include <float.h>

#include "gsl/gsl_rng.h"
#include "headers.h"
#include "mdp.h"

#include "sarsalambdaagent.h"
#include "tdlambdaagent.h"
#include "tdlambdanstepagent.h"
#include "tdnstep.h"
#include "sarsalambdatruncagent.h"

using namespace std;


void options(){

  cout << "Usage:\n";
  cout << "mdp-evaluate\n"; 
  cout << "\t[--s s]\n";
  cout << "\t[--p p]\n";
  cout << "\t[--chi chi]\n";
  cout << "\t[--w w]\n";
  cout << "\t[--sigma sigma]\n"; 
  cout << "\t[--method random | sarsa_lambda_alphainit_epsinit_initWeight | expsarsa_lambda_alphainit_epsinit_initWeight | greedygq_lambda_alphainit_epsinit_initWeight | qlearning_lambda_alphainit_epsinit_initWeight | ce_generations_evalepisodes | cmaes_generations_evalepisodes | ga_generations_evalepisodes | rwg_evalepisodes | transfer_lambda_alphaInit_epsInit_initWeight_generations_evalEpisodes]\n";
  cout << "\t[--totalEpisodes totalEpisodes]\n";
  cout << "\t[--displayInterval displayInterval]\n";
  cout << "\t[--randomSeed randomSeed]\n";
  cout << "\t[--outFile outFile]\n";
}


//  Read command line arguments, and set the ones that are passed (the others remain default.)
bool setRunParameters(int argc, char *argv[], int &s, double &p, double &chi, int &w, double &sigma, string &method, unsigned long int &totalEpisodes, unsigned long int &displayInterval, int &randomSeed, string &outFileName){

  int ctr = 1;
  while(ctr < argc){

    cout << string(argv[ctr]) << "\n";

    if(string(argv[ctr]) == "--help"){
      return false;//This should print options and exit.
    }
    else if(string(argv[ctr]) == "--s"){
      if(ctr == (argc - 1)){
	return false;
      }
      s = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--p"){
      if(ctr == (argc - 1)){
	return false;
      }
      p = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--chi"){
      if(ctr == (argc - 1)){
	return false;
      }
      chi = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--w"){
      if(ctr == (argc - 1)){
	return false;
      }
      w = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--sigma"){
      if(ctr == (argc - 1)){
	return false;
      }
      sigma = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--method"){
      if(ctr == (argc - 1)){
	return false;
      }
      method = argv[ctr + 1];
      ctr++;
    }
    else if(string(argv[ctr]) == "--totalEpisodes"){
      if(ctr == (argc - 1)){
	return false;
      }
      totalEpisodes = atol(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--displayInterval"){
      if(ctr == (argc - 1)){
	return false;
      }
      displayInterval = atol(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--randomSeed"){
      if(ctr == (argc - 1)){
	return false;
      }
      randomSeed = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--outFile"){
      if(ctr == (argc - 1)){
	return false;
      }
      outFileName = string(argv[ctr + 1]);
      ctr++;
    }
    else{
      return false;
    }

    ctr++;
  }

  return true;
}


// Run agent on m for numEpisodes, and compute average reward. No discounting.
double evaluate(MDP *m, Agent *agent, const int &numEpisodes, vector<double> stationary_dist){
  int x = 1;
  vector<double> features;
  vector <Cell> MDP_cells = m->cell;

  if (x == 1) {
	  vector < vector<double> >weights = agent->getWeightsFull();
	  double SSE = 0;
	  int numStates = 0;
	  //std :: cout << "*************************";
	  for (int index = 0; index < m->getNumStates(); index ++) {
		  features = MDP_cells[index].featureValue;
		  if (MDP_cells[index].isTerminal) {
			  continue;
		  }
		  numStates ++;
		  double optimalValue =  MDP_cells[index].maximalValue;
		  double approxValue = 0;
		  int action = MDP_cells[index].maximalAction;
		  for(int f = 0; f < m->getNumFeatures(); f++){
			  //std::cout << "after " <<  weights.size()<< "\n";
			  approxValue += weights[action][f] * features[f];
		  }
		  SSE += pow(optimalValue-approxValue,2) * stationary_dist[index];
		  //std :: cout << optimalValue << " " << approxValue << "\n";
	  }
	  //std :: cout << "*************************";
	  //std:: cout << SSE << " " << (m->getNumStates()) << "\n";

	  return sqrt((SSE));
  } else {
  double totalReward = 0;
  
  m->reset();

  for(int i = 0; i < numEpisodes; i++){

    bool term;
    do{

      features = m->getFeatures();
      //CHANGED
      int cellnumber = -1;
      int numcells = 0;

          for (int indexx = 0; indexx < MDP_cells.size(); indexx++) {
          	for (int indexx1 = 0; indexx1 < m->getNumFeatures(); indexx1++) {
          		if (features[indexx1] != MDP_cells[indexx].featureValue[indexx1]) {
          			break;
          		}
          		if (indexx1 == m->getNumFeatures()-1) {
          			cellnumber = indexx;
          			numcells++;
          		}
          	}
          }
          std::cout << numcells;
          features.push_back((double)MDP_cells[cellnumber].maximalAction);
          //CHANGED
      int action = agent->takeBestAction(features);
      term = m->takeAction(action);
      double r = m->getLastReward();
      totalReward += r;
    }
    while(term == false);

    //cout << "Episode " << i << " Total Reward: " << totalReward << "\n";
  }

  double averageReward = totalReward / numEpisodes;
  
  return averageReward;
  }
}


void test(MDP *m){

  char ac;
  do{
    
    m->display(DISPLAY_FEATURES);
    m->display(DISPLAY_TERMINAL);
    m->display(DISPLAY_REWARDS);
    m->display(DISPLAY_MAXIMAL_VALUES);
    m->display(DISPLAY_MINIMAL_VALUES);
    m->display(DISPLAY_MAXIMAL_ACTIONS);
    m->display(DISPLAY_MINIMAL_ACTIONS);

    cout << "Enter action n/e: ";
    cin >> ac;
    
    if(ac == 'n'){
      m->takeAction(ACTION_NORTH);
    }
    else if(ac == 'e'){
      m->takeAction(ACTION_EAST);
    }
  }
  while(ac == 'n' || ac == 's' || ac == 'e' || ac == 'w');
}


int main(int argc, char *argv[]){
  
  // Default parameter values.
  int s = 5;
  double p = 0.1;
  double chi = 0.5;
  int w = 1;
  double sigma = 0;
  string method = "sarsa_0_1.0_1.0";
  unsigned long int totalEpisodes = 50000;
  unsigned long int displayInterval = 2000;
  int randomSeed = time(0);
  string outFileName = "";


  if(!(setRunParameters(argc, argv, s, p, chi, w, sigma, method, totalEpisodes, displayInterval, randomSeed, outFileName))){
    options();
    return 1;
  }

  cout << "s: " << s << "\n";
  cout << "p: " << p << "\n";
  cout << "chi: " << chi << "\n";
  cout << "w: " << w << "\n";
  cout << "sigma: " << sigma << "\n";
  cout << "method: " << method << "\n";
  cout << "total episodes: " << totalEpisodes << "\n";
  cout << "display interval: " << displayInterval << "\n";
  cout << "randomSeed: " << randomSeed << "\n";
  cout << "outFileName: " << outFileName << "\n";

  // Start train and test MDP's with same parameters and random seed.
  MDP *trainMDP = new MDP(s, p, chi, w, sigma, randomSeed);
  MDP *testMDP = new MDP(s, p, chi, w, sigma, randomSeed);
  double minimalValue = testMDP->getMinimalValue();
  double maximalValue = testMDP->getMaximalValue();
  double randomValue = testMDP->getValueRandomPolicy();

  //cout << "train Maximal Value: " << trainMDP->getMaximalValue() << "\n";
  //cout << "train Minimal Value: " << trainMDP->getMinimalValue() << "\n";
  //cout << "train Random Value: " << trainMDP->getValueRandomPolicy() << "\n";
  //cout << "test Maximal Value: " << testMDP->getMaximalValue() << "\n";
  //cout << "test Minimal Value: " << testMDP->getMinimalValue() << "\n";
  //cout << "test Random Value: " << testMDP->getValueRandomPolicy() << "\n";
  //  test(testMDP);
  //test(trainMDP);
  //  return 1;

  
  // Initialise agent policy.
  Agent *agent;
  if(method.find("sarsa") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;
    double initWeight = 0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  pos = method.find("_", pos + 1);
	  if(pos != string::npos){
	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  }
	  else{
	    options(); return 1;
	  }
	}
	else{
	  options(); return 1;
	}
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }
    agent = new SarsaLambdaAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed);

  }
  else if(method.find("tdlambda") == 0){

      double lambda = 0;
      double alphaInit = 1.0;
      double epsInit = 1.0;
      double initWeight = 0;

      unsigned int pos = method.find("_");
      if(pos != string::npos){
        lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
        pos = method.find("_", pos + 1);
        if(pos != string::npos){
  	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
  	pos = method.find("_", pos + 1);
  	if(pos != string::npos){
  	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
  	  pos = method.find("_", pos + 1);
  	  if(pos != string::npos){
  	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
  	  }
  	  else{
  	    options(); return 1;
  	  }
  	}
  	else{
  	  options(); return 1;
  	}
        }
        else{
  	options(); return 1;
        }
      }
      else{
        options(); return 1;
      }
      agent = new TDLambdaAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed);

    }
  else if(method.find("tdnsteplambda") == 0){

        double lambda = 0;
        double alphaInit = 1.0;
        double epsInit = 1.0;
        double initWeight = 0;

        unsigned int pos = method.find("_");
        if(pos != string::npos){
          lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
          pos = method.find("_", pos + 1);
          if(pos != string::npos){
    	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
    	pos = method.find("_", pos + 1);
    	if(pos != string::npos){
    	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
    	  pos = method.find("_", pos + 1);
    	  if(pos != string::npos){
    	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
    	  }
    	  else{
    	    options(); return 1;
    	  }
    	}
    	else{
    	  options(); return 1;
    	}
          }
          else{
    	options(); return 1;
          }
        }
        else{
          options(); return 1;
        }
        agent = new TDLambdaNStepAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed, (int)epsInit);

      }
  else if(method.find("tdnstep") == 0){

          double lambda = 0;
          double alphaInit = 1.0;
          double epsInit = 1.0;
          double initWeight = 0;

          unsigned int pos = method.find("_");
          if(pos != string::npos){
            lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
            pos = method.find("_", pos + 1);
            if(pos != string::npos){
      	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      	pos = method.find("_", pos + 1);
      	if(pos != string::npos){
      	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      	  pos = method.find("_", pos + 1);
      	  if(pos != string::npos){
      	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      	  }
      	  else{
      	    options(); return 1;
      	  }
      	}
      	else{
      	  options(); return 1;
      	}
            }
            else{
      	options(); return 1;
            }
          }
          else{
            options(); return 1;
          }
          agent = new TDNStep(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed, (int)epsInit);

        }else if(method.find("truesarsa") == 0){

        	double lambda = 0;
        	      double alphaInit = 1.0;
        	      double epsInit = 1.0;
        	      double initWeight = 0;

        	      unsigned int pos = method.find("_");
        	      if(pos != string::npos){
        	        lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
        	        pos = method.find("_", pos + 1);
        	        if(pos != string::npos){
        	  	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
        	  	pos = method.find("_", pos + 1);
        	  	if(pos != string::npos){
        	  	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
        	  	  pos = method.find("_", pos + 1);
        	  	  if(pos != string::npos){
        	  	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
        	  	  }
        	  	  else{
        	  	    options(); return 1;
        	  	  }
        	  	}
        	  	else{
        	  	  options(); return 1;
        	  	}
        	        }
        	        else{
        	  	options(); return 1;
        	        }
        	      }
        	      else{
        	        options(); return 1;
        	      }
        	      agent = new SarsaLambdaTruncAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed);
          }

  else{
    options();
    return 1;
  }


  // How many episodes per evaluation?
  int numEvalEpisodes = 1000;
  /*CHANGED*/
  vector<Cell> MDP_cells = testMDP->cell;
  vector <double> value;
  for (int indexx = 0; indexx < testMDP->getNumStates(); indexx++) {
	  value.push_back(MDP_cells[indexx].maximalValue);
  }
  /*CHANGED*/
  // Points (number of training episodes) after which MDP should be evaluated.
  set<int> points;

  unsigned long int nextPoint = 0;
  while(nextPoint <= totalEpisodes){
    points.insert(nextPoint);
    nextPoint += displayInterval;
  }

  for (int i = 0; i < 1000*testMDP->getNumStates(); i++) {
	  bool term = false;
	  while ( ! term) {
		  int action = (double)testMDP->getActionOpt();
		  term = testMDP->takeAction(action);
	  }
  }
  MDP_cells = testMDP->cell;
  vector <double> MDP_weights;
  double total = 0;
  for (int i = 0; i< testMDP->getNumStates(); i++) {
	  total +=MDP_cells[i].visits;
	  //cout << MDP_cells[i].visits << " ";
  }
  cout << "\n WEIGHTS \n";
  for (int i = 0; i< testMDP->getNumStates(); i++) {
  	  MDP_weights.push_back(MDP_cells[i].visits/total);
  }

  for (int i = 0; i< testMDP->getNumStates(); i++) {
	//  std :: cout << MDP_weights[i] << " ";
  }
  double normal = 0;
  for (int i = 0; i< testMDP->getNumStates(); i++) {
	  normal+= MDP_weights[i];
  }
//  std :: cout << total << " " << normal << " ";

  // Write output.
  stringstream output(stringstream::in | stringstream::out);
  output << "# start output\n";

  output << "# start values\n";

  for(unsigned int e = 0; e <= totalEpisodes; e++){
		  
    if(points.find(e) != points.end()){

      double val = evaluate(testMDP, agent, numEvalEpisodes, MDP_weights);
      //val = (val - minimalValue) / (maximalValue - minimalValue);
      //CHANGEDval = (val - randomValue) / (maximalValue - randomValue);
      output << e << "\t" << val << "\n";
      //cout << e << "\t" << val << "\n";
    }

    vector<double> state;
    int action;
    bool term;
    double reward;

    term = false;
    state = trainMDP->getFeatures();
    /*CHANGED*/
    /*
    int cellnumber = -1;
    int count = 0;

    for (int indexx = 0; indexx < MDP_cells.size(); indexx++) {
    	for (int indexx1 = 0; indexx1 < testMDP->getNumFeatures(); indexx1++) {
    		if (state[indexx1] != MDP_cells[indexx].featureValue[indexx1]) {
    			break;
    		}
    		if (indexx1 == testMDP->getNumFeatures()-1) {
    			cellnumber = indexx;
    			count ++;
    		}
    	}
    }
    if (count != 1)
    	std::cout << count << " " << cellnumber << "\n";
    state.push_back((double)MDP_cells[cellnumber].maximalAction);*/
    /*CHANGED*/
    while(!term){
      state.push_back((double)trainMDP->getActionOpt());
      action = agent->takeAction(state);
      term = trainMDP->takeAction(action);
      reward = trainMDP->getLastReward();
      state = trainMDP->getFeatures();
      agent->update(reward, state, term);
    }
    
  }

  output << "# end values\n";

  output << "# end output\n";

  // If output file is specified, write to it; otherwise write to console.
  if(outFileName.length() > 0){
    fstream file;
    file.open(outFileName.c_str(), ios::out);
    file << output.str();
    file.close();
  }
  else{
    cout << output.str();
  }


  // Deallocate memory.
  delete agent;
  delete trainMDP;
  delete testMDP;

  return 0;
}


