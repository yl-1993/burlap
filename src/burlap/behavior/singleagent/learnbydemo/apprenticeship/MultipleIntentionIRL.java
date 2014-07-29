package burlap.behavior.singleagent.learnbydemo.apprenticeship;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.QValue;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.learnbydemo.apprenticeship.requests.EMMLIRLRequest;
import burlap.behavior.singleagent.learnbydemo.apprenticeship.requests.MLIRLRequest;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.commonpolicies.BoltzmannQPolicy;
import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.Action;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;

public class MultipleIntentionIRL {


	public static Policy runMLIRL(MLIRLRequest request) {
		
		// Get all required items from the request object
		Domain domain = request.getDomain();
		StateToFeatureVectorGenerator featureGenerator = request.getFeatureGenerator();
		TerminalFunction terminalFunction = request.getTerminalFunction();
		Double gamma = request.getGamma();
		State startState = request.getStartStateGenerator().generateState();
		StateHashFactory hashFactory = request.getStateHashFactory();
		double beta = request.getBeta();
		double epsilon = request.getEpsilon();
		List<Double> trajectoryWeights = request.getTrajectoryWeights();
		List<EpisodeAnalysis> trajectories = request.getExpertEpisodes();
		OOMDPPlanner planner = request.getPlanner();
		int numIterations = request.getMaxIterations();
		int featureVectorLength = request.getFeatureVectorLength();
		
		// Initialize choose random set of reward weights
		FeatureWeights featureWeights = 
				getRandomRewardWeights(featureVectorLength);
		
		FeatureWeights lastFeatureWeights = featureWeights;
		BoltzmannPolicySum bPolicySum = new BoltzmannPolicySum(
				(QComputablePlanner)planner, featureGenerator, featureWeights, beta, gamma);
		
		// TODO set a realistic alpha
		double alpha = 1.0;
		double squareError = 0.0;
		// for t = 1 to M do
		for (int i = 0; i < numIterations; i++) {
			// Compute new feature weights through gradient ascent, theta_{t+1} = theta_t + alpha * G(L)
			lastFeatureWeights = computeNewFeatureWeightsViaGradientAscent
					(featureWeights, trajectoryWeights, bPolicySum,
							 trajectories, startState, domain, hashFactory, alpha);
			alpha *= 0.99;
			// Compute error
			squareError = computeSquareError(lastFeatureWeights, featureWeights);
			// Update
			featureWeights = lastFeatureWeights;
			// Check Convergence
			if(squareError < epsilon){
				break;
			}
			// Compute log likelihood of data (L), L is supposed to be larger and larger
			double logLikelihood = computeLogLikelihood(trajectoryWeights, bPolicySum, trajectories);
			System.out.println("L: "+logLikelihood+"\n");
		}
		
		// Compute new reward functions using new feature weights
		RewardFunction rewardFunction = 
				ApprenticeshipLearning.generateRewardFunction(featureGenerator, featureWeights);
		
		// Reset planner to accommodate new reward function
		planner.resetPlannerResults();
		planner.plannerInit(domain, rewardFunction, terminalFunction, gamma , hashFactory);
		
		// From feature weights, compute Q_theta_t, and generate a new policy from the Q function
		Policy policy = computePolicyFromRewardWeights(planner, featureWeights, featureGenerator, startState, beta);
		
		// Evaluate the behavior of the new policy
		String res = policy.evaluateBehavior(request.getStartStateGenerator().generateState(), rewardFunction, terminalFunction).getActionSequenceString();
		System.out.println("ActionSequence: "+res+"\n");
		
		System.out.println("mlirl finish!");
		
		return policy;
	}

	
	public static BoltzmannPolicySum runMLIRL(MLIRLRequest request, FeatureWeights featureWeights){
		// Get all required items from the request object
		Domain domain = request.getDomain();
		StateToFeatureVectorGenerator featureGenerator = request.getFeatureGenerator();
		TerminalFunction terminalFunction = request.getTerminalFunction();
		Double gamma = request.getGamma();
		State startState = request.getStartStateGenerator().generateState();
		StateHashFactory hashFactory = request.getStateHashFactory();
		double beta = request.getBeta();
		double epsilon = request.getEpsilon();
		List<Double> trajectoryWeights = request.getTrajectoryWeights();
		List<EpisodeAnalysis> trajectories = request.getExpertEpisodes();
		OOMDPPlanner planner = request.getPlanner();
		int numIterations = request.getMaxIterations();
		
		FeatureWeights lastFeatureWeights = new FeatureWeights(featureWeights);
		BoltzmannPolicySum bPolicySum = new BoltzmannPolicySum(
				(QComputablePlanner)planner, featureGenerator, featureWeights, beta, gamma);
		BoltzmannPolicySum lastBPolicySum = new BoltzmannPolicySum(bPolicySum);
		// TODO set a realistic alpha
		double alpha = 1.0;
		double squareError = 0.0;
		// for t = 1 to M do
		for (int i = 0; i < numIterations; i++) {
			// Compute new feature weights through gradient ascent, theta_{t+1} = theta_t + alpha * G(L)
			lastFeatureWeights = computeNewFeatureWeightsViaGradientAscent
					(featureWeights, trajectoryWeights, lastBPolicySum,
							 trajectories, startState, domain, hashFactory, alpha);
			alpha *= 0.99;
			if(isValid(lastFeatureWeights.getWeights())) {
				// Compute error
				squareError = computeSquareError(lastFeatureWeights, featureWeights);
				// Update
				featureWeights = lastFeatureWeights;
				bPolicySum = lastBPolicySum;
			}
			else{
				break;
			}
			// Check Convergence
			if(squareError < epsilon){
				break;
			}
			// Compute log likelihood of data (L), L is supposed to be larger and larger
			double logLikelihood = computeLogLikelihood(trajectoryWeights, bPolicySum, trajectories);
			System.out.println("L: "+logLikelihood+"\n");
		}
		
//		// Compute new reward functions using new feature weights
//		RewardFunction rewardFunction = 
//				ApprenticeshipLearning.generateRewardFunction(featureGenerator, featureWeights);
//		
//		// Reset planner to accommodate new reward function
//		planner.resetPlannerResults();
//		planner.plannerInit(domain, rewardFunction, terminalFunction, gamma , hashFactory);
//		
//		// From feature weights, compute Q_theta_t, and generate a new policy from the Q function
//		Policy policy = computePolicyFromRewardWeights(planner, featureWeights, featureGenerator, startState, beta);
//		
//		// Evaluate the behavior of the new policy
//		String res = policy.evaluateBehavior(request.getStartStateGenerator().generateState(), rewardFunction, terminalFunction).getActionSequenceString();
//		System.out.println("ActionSequence: "+res+"\n");
//		
//		System.out.println("mlirl finish!");
		
		return bPolicySum;
	}
	
	
	public static List<double[]> runEM(EMMLIRLRequest request) {
		
		// Get all required items for EM
		int numberClusters = request.getNumberClusters();
		List<EpisodeAnalysis> episodes = request.getExpertEpisodes();
		int numberTrajectories = episodes.size();
		int featureWeightLength = request.getFeatureVectorLength();
		int numberIterations = request.getMaxIterations();
		numberIterations = 3;
		List<double[]> featureWeightClusters =
				new ArrayList<double[]>(numberClusters);
		
		// Randomly initialize the prior probabilities of each cluster
		FeatureWeights clusterPriors = 
				getRandomRewardWeights(numberClusters);
		double[] clusterPriorProbabilities = clusterPriors.getWeights();
		
		// Randomly initialize the feature weights of each cluster
		for (int i = 0; i < numberClusters; i++) {
			FeatureWeights weights = 
					getRandomRewardWeights(featureWeightLength);
			featureWeightClusters.add(weights.getWeights());
		}
		
		// Suppose every cluster has the same probability
		List<double[]> trajectoryInClustersProbabilities = new ArrayList<>(numberTrajectories);
		double[] trajectoryInClustersProbability = new double[numberClusters];
		for (int j = 0; j < numberClusters; j++) {
			trajectoryInClustersProbability[j] = 1.0/(numberClusters*numberTrajectories);
		}
		for (int i = 0; i < numberTrajectories; i++) {			
			trajectoryInClustersProbabilities.add(trajectoryInClustersProbability);
		}
		
		// Generate policies from initial values
		// run MLIRL firstly
		List<BoltzmannPolicySum> bPolicySums = computeClusterPolicies(request, featureWeightClusters, trajectoryInClustersProbabilities);
		
		// repeat until target number of iterations completed
		for (int i = 0; i < numberIterations; i++) {
			stepEM(request, episodes, featureWeightClusters, clusterPriorProbabilities, bPolicySums);
		}
		
		trajectoryInClustersProbabilities = computeTrajectoryInClusterProbabilities(
				bPolicySums, episodes, clusterPriorProbabilities);
		
		for (int i = 0; i < trajectoryInClustersProbabilities.size(); i++) {
			double[] tmp = trajectoryInClustersProbabilities.get(i);
			int index = 0;
			double m = Double.MIN_VALUE;
			for (int j = 0; j < tmp.length; j++) {
				if(tmp[j] > m){
					m = tmp[j];
					index = j;
				}
				System.out.println(tmp[j]+",");
			}
			System.out.println(index);
			System.out.println("\n");
		}
		
		return featureWeightClusters;
	}

	/**
	 * Generates a random set of reward weights of given length
	 * @param length
	 * @return
	 */
	protected static FeatureWeights getRandomRewardWeights(int length) {
		Random random = new Random();
		double[] weights = new double[length];
		double sum = 0;
		for (int i = 0; i < length; i++) {
			weights[i] = -1 + random.nextDouble() * (1 - (-1));
			sum += weights[i];
		}
		for (int i = 0; i < length; i++) {
			weights[i] /= sum;
		}
		return new FeatureWeights(weights);
	}
	
	/**
	 * Given reward feature weights this method computes a new BoltzmannQPolicy
	 * @param planner
	 * @param featureWeights
	 * @param featureGenerator
	 * @param startState
	 * @param beta
	 * @return A BoltzmannQPolicy generated from the feature weights
	 */
	// TODO protected or public???
	public static Policy computePolicyFromRewardWeights(OOMDPPlanner planner, 
			FeatureWeights featureWeights,
			StateToFeatureVectorGenerator featureGenerator, State startState, double beta) {

		RewardFunction rewardFunction = 
				ApprenticeshipLearning.generateRewardFunction(featureGenerator, featureWeights);
		planner.setRf(rewardFunction);
		planner.planFromState(startState);
		
		return new BoltzmannQPolicy((QComputablePlanner)planner, beta);
	}
	
	/**
	 * Computes L = sum_i w_i sum_{(s,a) \in \xi_i} log(pi_theta_t(s,a))
	 * Computes the log likelihood of the trajectories given a policy and weights of each trajectory
	 * @param trajectoryWeights
	 * @param bPolicySum
	 * @param trajectories
	 * @return the log-likelihood of the trajectories
	 */
	protected static double computeLogLikelihood(
			List<Double> trajectoryWeights, 
			BoltzmannPolicySum bPolicySum, 
			List<EpisodeAnalysis> trajectories) {
		
		double logLikelihood = 0.0;
		
		for (int i = 0; i < trajectories.size(); i++) {
			
			EpisodeAnalysis episode = trajectories.get(i);
			List<GroundedAction> actions = episode.actionSequence;
			List<State> states = episode.stateSequence;
			List<Double> rewards = episode.rewardSequence;
			int trajectoryLength = Math.min(actions.size(), states.size());
			double sum = 0.0;
			double probability = 0.0;
			
			for (int j = 0; j < trajectoryLength; j++) {
				QValue qValue = new QValue(states.get(j),actions.get(j),rewards.get(j));
				probability = bPolicySum.getPi(states.get(j), qValue);
				if(probability == 0){
					probability = 1e-6;
				}
				sum += Math.log(probability);
			}
			
			logLikelihood += trajectoryWeights.get(i)*sum;
		}
		
		return logLikelihood;
	}
	
	// w, r, V_{i-1}, V'_{i-1} will not change for this iteration
	protected static double computeLogLikelihoodPrime(
			List<EpisodeAnalysis> trajectories, 
			BoltzmannPolicySum bPolicySum,
			List<Double> trajectoryWeights, 		
			FeatureWeights featureWeights,
			int feature){
		double lPrimeValue = 0.0;
		for (int i = 0; i < trajectories.size(); i++) {
			
			EpisodeAnalysis episode = trajectories.get(i);
			List<GroundedAction> actions = episode.actionSequence;
			List<State> states = episode.stateSequence;
			List<Double> rewards = episode.rewardSequence;
			int trajectoryLength = Math.min(actions.size(), states.size());
			double sum = 0.0;
			// sum_j (1 / Pi(s,a))*(d Pi(s,a) / d feature)
			double stepValue = 0.0;
			for (int j = 0; j < trajectoryLength; j++) {
				QValue qValue = new QValue(states.get(j),actions.get(j),rewards.get(j));
				stepValue = bPolicySum.getPi(states.get(j), qValue);
				if (stepValue != 0) {
					stepValue = (bPolicySum.getPiPrime(states.get(j), qValue, feature) / stepValue);
				}		
				if(Double.isNaN(stepValue)){
					System.out.println(bPolicySum.getPiPrime(states.get(j), qValue, feature) + "num NAN!!!");
					stepValue = 0;
				}
				sum += stepValue;
				// if sum is NAN, stop it and use weights generated by last iteration
				if(Double.isNaN(sum)){
					System.out.println("NAN!!!");
				}
			}
			lPrimeValue += trajectoryWeights.get(i) * sum;
		}
		
		return lPrimeValue;
	}
	
	/**
	 * Computes theta_{t+1} = theta_t + alpha * G(L)
	 * Computes new feature weights by taking a small step in the positive step of the gradient
	 * @param featureWeights
	 * @param trajectoryWeights
	 * @param policy
	 * @param episodes
	 * @return New feature weights
	 */
	protected static FeatureWeights computeNewFeatureWeightsViaGradientAscent(
			FeatureWeights featureWeights, List<Double> trajectoryWeights, 
			BoltzmannPolicySum bPolicySum,  List<EpisodeAnalysis> episodes, 
			State startState, Domain domain,
			StateHashFactory hashFactory, double alpha) {
		int kIterationNum = 1;
		double lPrimeValue = 0.0;
		double[] weights = featureWeights.getWeights();
		double[] dWeights = new double[weights.length];
		// weights and rewards will not change during one iteration
		// compute new rewards
		updateRewards(episodes, bPolicySum, startState, domain, hashFactory);
		// k iterations
		for (int k = 0; k < kIterationNum; k++) {
			bPolicySum.clearAllValuesExceptV();
			updateLastV(bPolicySum, startState, domain, hashFactory);
			updatePI(bPolicySum, startState, domain, hashFactory);
			updateV(bPolicySum, startState, domain, hashFactory);
			// PI, V will not change during the computation of LikelihoodPrime
			// d V / d w_i, dL / d w_i
			for (int i = 0; i < weights.length; i++) {	
				// compute new V and V prime
				updateLastVPrime(bPolicySum, startState, domain, hashFactory,i);
				updatePIPrime(bPolicySum, startState, domain, hashFactory, i);
				updateVPrime(bPolicySum, startState, domain, hashFactory, i);
				lPrimeValue = computeLogLikelihoodPrime(
						episodes, bPolicySum, trajectoryWeights, featureWeights, i);
				dWeights[i] = (alpha * lPrimeValue); 				
			}
		}
		// Update w^{t+1}, r_{t+1}
		for (int i = 0; i < weights.length; i++) {
			weights[i] += dWeights[i];
			System.out.println(weights[i]+",");
		}
		System.out.println("\n");
		// set new weights
		updateFeatureWeights(weights, bPolicySum);
		
		return new FeatureWeights(weights);
	}

	protected static void updateLastV(BoltzmannPolicySum bPolicySum, 
			State startState, Domain domain,
			StateHashFactory hashFactory) {
		List <State> allStates = StateReachability.getReachableStates(startState, 
			(SADomain)domain, hashFactory);
		for (State state : allStates) {
			bPolicySum.updateLastValue(state);
		}
	}
	
	protected static void updateLastVPrime(BoltzmannPolicySum bPolicySum, 
			State startState, Domain domain,
			StateHashFactory hashFactory, int feature) {
		List <State> allStates = StateReachability.getReachableStates(startState, 
			(SADomain)domain, hashFactory);
		for (State state : allStates) {
			bPolicySum.updateLastValuePrime(state, feature);
		}
	}

	protected static void updateV(BoltzmannPolicySum bPolicySum, 
			State startState, Domain domain,
			StateHashFactory hashFactory) {
		List <State> allStates = StateReachability.getReachableStates(startState, 
			(SADomain)domain, hashFactory);
		for (State state : allStates) {
			bPolicySum.getValue(state);
		}
	}
	
	protected static void updateVPrime(BoltzmannPolicySum bPolicySum, 
			State startState, Domain domain,
			StateHashFactory hashFactory, int feature) {
		List <State> allStates = StateReachability.getReachableStates(startState, 
			(SADomain)domain, hashFactory);
		for (State state : allStates) {
			bPolicySum.getValuePrime(state, feature);
		}
	}
	
	protected static void updatePI(
			BoltzmannPolicySum bPolicySum, State startState, Domain domain,
			StateHashFactory hashFactory) {
		// The computation of PI will include the computation of Q
		// compute pi and Q of all states and actions
		List <State> allStates = StateReachability.getReachableStates(startState, 
				(SADomain)domain, hashFactory);
		for (State state : allStates) {
			List<GroundedAction> allActions = Action.getAllApplicableGroundedActionsFromActionList(domain.getActions(),state);
			for (GroundedAction groundedAction : allActions) {
				QValue qValue = new QValue(state, groundedAction, 0.0);
				bPolicySum.getPi(state, qValue);
			}
		}
	}
	
	protected static void updatePIPrime(
			BoltzmannPolicySum bPolicySum, State startState, Domain domain,
			StateHashFactory hashFactory, int feature) {
		// The computation of PI will include the computation of Q
		// compute pi and Q of all states and actions
		List <State> allStates = StateReachability.getReachableStates(startState, 
				(SADomain)domain, hashFactory);
		for (State state : allStates) {
			List<GroundedAction> allActions = Action.getAllApplicableGroundedActionsFromActionList(domain.getActions(),state);
			for (GroundedAction groundedAction : allActions) {
				QValue qValue = new QValue(state, groundedAction, 0.0);
				bPolicySum.getPiPrime(state, qValue, feature);
			}
		}
	}
	
	// the rewards of states maybe computed more than one time
	protected static void updateRewards(List<EpisodeAnalysis> trajectories, 
			BoltzmannPolicySum bPolicySum, State startState, Domain domain,
			StateHashFactory hashFactory) {
		List <State> allStates = StateReachability.getReachableStates(startState, 
				(SADomain)domain, hashFactory);
		for (State state : allStates) {
				bPolicySum.updateRewardValues(state);
		}
	}

	protected static void updateFeatureWeights(double[] weights, BoltzmannPolicySum bPolicySum) {
		bPolicySum.updateFeatureWeights(weights);
	}
	
	protected static double computeSquareError(FeatureWeights lastFeatureWeights, FeatureWeights curFeatureWeights) {
		double squareError = 0.0;
		double[] lastWeights = lastFeatureWeights.getWeights();
		double[] curWeights = curFeatureWeights.getWeights();
		int len = Math.min(lastWeights.length, curWeights.length);
		for (int i = 0; i < len; i++) {
			squareError += ((lastWeights[i] - curWeights[i])*(lastWeights[i] - curWeights[i]));
		}
		return squareError;
	}
	
	/**
	 * A single step in the EM algorithm
	 * @param request
	 * @param episodes
	 * @param featureWeightClusters
	 * @param clusterPriorProbabilities
	 * @param policies
	 */
	protected static void stepEM(EMMLIRLRequest request, List<EpisodeAnalysis> episodes, 
			List<double[]> featureWeightClusters,
			double[] clusterPriorProbabilities, 
			List<BoltzmannPolicySum> bPolicySums) {
		
		// E Step
		// Compute z_ij = prod pi * rho / Z
		List<double[]> trajectoryInClustersProbabilities = computeTrajectoryInClusterProbabilities(
				bPolicySums, episodes, clusterPriorProbabilities);
		
		// M Step
		// Compute rho_l = sum z_il / N
		clusterPriorProbabilities = computePriors(trajectoryInClustersProbabilities);
		
		// Compute theta_l via MLIRL and generate appropriate policies
		bPolicySums = computeClusterPolicies(request, featureWeightClusters, trajectoryInClustersProbabilities);
	}
	
	/**
	 * Computes z_ij = prod_{(s,a) \in \xi_i} pi_{theta_j}(s,a) * \rho_j / Z
	 * This finds the probability that trajectory i is in cluster j by finding the product of policy 
	 * probabilities from the trajectories state-action pairs and normalizing
	 * @param policies List of policies used associated with each cluster
	 * @param trajectories List of expert trajectories
	 * @param clusterPriorProbabilities Prior probabilities of each cluster
	 * @return The probabilities of each trajectory belonging to each cluster.
	 */
	protected static List<double[]> computeTrajectoryInClusterProbabilities(
			List<BoltzmannPolicySum> bPolicySums, List<EpisodeAnalysis> trajectories, 
			double[] clusterPriorProbabilities) {
		
		List<double[]> trajectoryInClusterProbabilities = 
				new ArrayList<double[]>(trajectories.size());
		int numberClusters = clusterPriorProbabilities.length;
		EpisodeAnalysis episode;
		BoltzmannPolicySum bPolicySum;
		double prior;
		double sum;
		
		for (int i = 0; i < trajectories.size(); i++) {
			double[] probabilities = new double[numberClusters];
			sum = 0;
			for (int j = 0; j < numberClusters; j++) {
				probabilities[j] = 0.0;
				episode = trajectories.get(i);
				bPolicySum = bPolicySums.get(j);
				prior = clusterPriorProbabilities[j];
				probabilities[j] = computeTrajectoryInClusterProbability(episode, bPolicySum, prior);
				sum += probabilities[j];
			}
			
			for (int j = 0; j < numberClusters; j++) {
				probabilities[j] /= sum;
			}
			
			trajectoryInClusterProbabilities.add(probabilities);
		}
		return trajectoryInClusterProbabilities;
	}
	
	/**
	 * Computes prod_{(s,a) \in \xi} pi_theta(s,a) * \rho
	 * This computes the unnormalized probability of a trajectory in a given cluster, the inner loop of the
	 * above method
	 * @param trajectory
	 * @param policy
	 * @param prior
	 * @return An unnormalized probability that the trajectory belongs to this cluster
	 */
	protected static double computeTrajectoryInClusterProbability(EpisodeAnalysis trajectory, 
			BoltzmannPolicySum bPolicySum, double prior) {
		double product = prior;
		List<GroundedAction> actions = trajectory.actionSequence;
		List<State> states = trajectory.stateSequence;
		List<Double> rewards = trajectory.rewardSequence;
		int trajectoryLength = Math.min(actions.size(), states.size());
		
		State state;
		GroundedAction action;
		Double reward;
		for (int i = 0; i < trajectoryLength; i++) {
			state = states.get(i);
			action = actions.get(i);
			reward = rewards.get(i);
			//product *= policy.getProbOfAction(state, action); //getPi
			QValue qValue = new QValue(state, action, reward);
			product *= bPolicySum.getPi(state, qValue);
		}
		
		return product;	
	}
	
	/**
	 * Computes \rho_l = \sum_i z_il / N
	 * This computes the prior probabilities of each cluster by summing over all trajectories the probability
	 * of each trajectory in the cluster.
	 * @param trajectoryInClustersProbabilities The probabilities of each trajectory belonging to each cluster
	 * @return The prior probabilities of each cluster
	 */
	protected static double[] computePriors(List<double[]> trajectoryInClustersProbabilities) {
		
		int numberClusters = trajectoryInClustersProbabilities.size();
		int numberTrajectories = trajectoryInClustersProbabilities.get(0).length;
		double[] priors = new double[numberClusters];
		
		double sum;
		double[] trajectoryInClusterProbabilities;
		for (int i = 0; i < numberClusters; i++) {
			sum = 0;
			trajectoryInClusterProbabilities = trajectoryInClustersProbabilities.get(i);
			for (int j = 0; j < numberTrajectories; j++) {
				sum += trajectoryInClusterProbabilities[j];
			}
			priors[i] = sum / numberTrajectories;
		}
		
		return priors;
	}
	
	/**
	 * Generates a policies for each cluster given probabilities of the trajectories belonging to each cluster
	 * It computes this via MLIRL
	 * @param request
	 * @param featureWeightClusters
	 * @param trajectoryInClustersProbabilities (number of trajectories * number of cluster)
	 * @return The policies generated for each cluster
	 */
	protected static List<BoltzmannPolicySum> computeClusterPolicies(EMMLIRLRequest request,
			List<double[]> featureWeightClusters,
			List<double[]> trajectoryInClustersProbabilities) {
		
		int numberClusters = featureWeightClusters.size();
		EMMLIRLRequest newRequest = new EMMLIRLRequest(request);
		List<BoltzmannPolicySum> bPolicySums = new ArrayList<BoltzmannPolicySum>(numberClusters);
		BoltzmannPolicySum bPolicySum;
		double[] trajectoryWeights = new double[trajectoryInClustersProbabilities.size()];
		
		for (int i = 0; i < numberClusters; i++) {
			for (int j = 0; j < trajectoryInClustersProbabilities.size(); j++) {
				trajectoryWeights[j] = trajectoryInClustersProbabilities.get(j)[i];
			}
			newRequest.setTrajectoryWeights(trajectoryWeights);
			FeatureWeights featureWeights = new FeatureWeights(featureWeightClusters.get(i));
			bPolicySum = runMLIRL(newRequest, featureWeights);
			bPolicySums.add(bPolicySum);
		}
		
		return bPolicySums;
	}
	
	protected static boolean isValid(double[] array) {
		for (int i = 0; i < array.length; i++) {
			if(Double.isNaN(array[i]))
				return false;
		}
		return true;
	}
	
}
