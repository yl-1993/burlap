package burlap.behavior.singleagent.learnbydemo.apprenticeship;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.QValue;

import org.apache.commons.lang3.ArrayUtils;
import org.omg.CORBA.INTERNAL;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.learnbydemo.apprenticeship.ApprenticeshipLearning.FeatureWeights;
import burlap.behavior.singleagent.learnbydemo.apprenticeship.requests.MLIRLRequest;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.commonpolicies.BoltzmannQPolicy;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
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
		List<Double> trajectoryWeights = request.getTrajectoryWeights();
		List<EpisodeAnalysis> trajectories = request.getExpertEpisodes();
		OOMDPPlanner planner = request.getPlanner();
		int numIterations = request.getMaxIterations();
		int featureVectorLength = request.getFeatureVectorLength();
		// Initialize choose random set of reward weights
		ApprenticeshipLearning.FeatureWeights featureWeights = 
				getRandomRewardWeights(featureVectorLength);
		
		// Jerry
		BoltzmannPolicySum bPolicySum = new BoltzmannPolicySum(
				(QComputablePlanner)planner, featureGenerator, featureWeights, beta, gamma);
		
		// TODO set a realistic alpha
		double alpha = 1.0;
		
		// for t = 1 to M do
		for (int i = 0; i < numIterations; i++) {
			// Compute new feature weights through gradient ascent, theta_{t+1} = theta_t + alpha * G(L)
			featureWeights = computeNewFeatureWeightsViaGradientAscent
					(featureWeights, trajectoryWeights, bPolicySum,
							 trajectories, startState, domain, hashFactory, alpha);
			alpha *= 0.99;
			// Compute log likelihood of data (L)
			//double logLikelihood = computeLogLikelihood(trajectoryWeights, policy, trajectories);
			//System.out.println("L: "+logLikelihood+"\n");
			
		}
//		double[] tmp = featureWeights.getWeights();
//		tmp[0] = 0.8553665098373343;
//		tmp[1] = 0.45023816863448645;
//		tmp[2] = -0.02462960847686731;
//		tmp[3] = 0.4420365715459394676;
//		tmp[4] = -0.02391545198949;
//		tmp[5] = -tmp[1];
//		tmp[6] = -tmp[2];
//		tmp[7] = -tmp[3];
//		tmp[8] = -tmp[4];
//		featureWeights = new ApprenticeshipLearning.FeatureWeights(tmp,1.0);
		
		// Compute new reward functions using new feature weights
		RewardFunction rewardFunction = 
				ApprenticeshipLearning.generateRewardFunction(featureGenerator, featureWeights);
		
		// Need to reset planner every time to accommodate new reward function
		planner.resetPlannerResults();
		planner.plannerInit(domain, rewardFunction, terminalFunction, gamma , hashFactory);
		
		// From feature weights, compute Q_theta_t, new reward function and generate a new policy from the Q function
		Policy policy = computePolicyFromRewardWeights(planner, featureWeights, featureGenerator, startState, beta);
		
		//String res = policy.evaluateBehavior(request.getStartStateGenerator().generateState(), rewardFunction, terminalFunction).getActionSequenceString();
		//System.out.println("res: "+res+"\n");
		
		System.out.println("mlirl finish!");
		
		return policy;
	}


	public static void runEM(MLIRLRequest request) {
		
		// Get all required items for EM
		int numberClusters = request.getNumberClusters();
		List<EpisodeAnalysis> episodes = request.getExpertEpisodes();
		int featureWeightLength = request.getFeatureVectorLength();
		int numberIterations = request.getMaxIterations();
		
		List<double[]> featureWeightClusters =
				new ArrayList<double[]>(numberClusters);
		
		// Randomly initialize the prior probabilities of each cluster
		ApprenticeshipLearning.FeatureWeights clusterPriors = 
				getRandomRewardWeights(numberClusters);
		double[] clusterPriorProbabilities = clusterPriors.getWeights();
		
		// Randomly initialize the feature weights of each cluster
		for (int i = 0; i < numberClusters; i++) {
			ApprenticeshipLearning.FeatureWeights weights = 
					getRandomRewardWeights(featureWeightLength);
			featureWeightClusters.add(weights.getWeights());
		}
		
		// Generate policies from initial values
		List<Policy> policies = computeClusterPolicies(request, featureWeightClusters);
		
		// repeat until target number of iterations completed
		for (int i = 0; i < numberIterations; i++) {
			stepEM(request, episodes, featureWeightClusters, clusterPriorProbabilities, policies);
		}
	}

	/**
	 * Generates a random set of reward weights of given length
	 * @param length
	 * @return
	 */
	protected static ApprenticeshipLearning.FeatureWeights getRandomRewardWeights(int length) {
		Random random = new Random();
		double[] weights = new double[length];
		double sum = 0;
		for (int i = 0; i < length; i++) {
			weights[i] = random.nextDouble();
			sum += weights[i];
		}
		for (int i = 0; i < length; i++) {
			weights[i] /= sum;
		}
		return new ApprenticeshipLearning.FeatureWeights(weights, 0.0);
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
	protected static Policy computePolicyFromRewardWeights(OOMDPPlanner planner, 
			ApprenticeshipLearning.FeatureWeights featureWeights,
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
	 * @param policy
	 * @param trajectories
	 * @return the log-likelihood of the trajectories
	 */
	protected static double computeLogLikelihood(
			List<Double> trajectoryWeights, Policy policy, List<EpisodeAnalysis> trajectories) {
		
		double logLikelihood = 0.0;
		
		for (int i = 0; i < trajectories.size(); i++) {
			
			EpisodeAnalysis episode = trajectories.get(i);
			List<GroundedAction> actions = episode.actionSequence;
			List<State> states = episode.stateSequence;
			int trajectoryLength = Math.min(actions.size(), states.size());
			double sum = 0.0;
			double probability = 0.0;
			
			for (int j = 0; j < trajectoryLength; j++) {
				GroundedAction ga = actions.get(j);
				State s = states.get(j);
				probability = policy.getProbOfAction(s, ga); 
				sum += Math.log(probability);
			}
			
			logLikelihood += trajectoryWeights.get(i)*sum;
		}
		
		return logLikelihood;
	}
	
	// Jerry
	// w, r, V_{i-1}, V'_{i-1} will not change for this iteration
	protected static double computeLogLikelihoodPrime(
			List<EpisodeAnalysis> trajectories, 
			BoltzmannPolicySum bPolicySum,
			List<Double> trajectoryWeights, 		
			ApprenticeshipLearning.FeatureWeights featureWeights,
			int feature){
		double lPrimeValue = 0.0;
		for (int i = 0; i < trajectories.size(); i++) {
			
			EpisodeAnalysis episode = trajectories.get(i);
			List<GroundedAction> actions = episode.actionSequence;
			List<State> states = episode.stateSequence;
			List<Double> rewards = episode.rewardSequence;
			int trajectoryLength = Math.min(actions.size(), states.size());
			double sum = 0.0;
			
			for (int j = 0; j < trajectoryLength; j++) {
				QValue qValue = new QValue(states.get(j),actions.get(j),rewards.get(j));
				// 
				sum += (bPolicySum.getPiPrime(states.get(j), qValue, feature)
						/ bPolicySum.getPi(states.get(j), qValue));	
				//
				//bPolicySum.getValuePrime(states.get(j), feature);
				//
//				System.out.println(states.get(j).toString());
//				System.out.println(actions.get(j).toString());
//				System.out.println(rewards.get(j).toString());
//				System.out.println(bPolicySum.getValue(states.get(j)));
				if(Double.isNaN(bPolicySum.getPiPrime(states.get(j), qValue, feature))){
					System.out.println("NAN!!!");
					//return 0.0;
				}
			}
			lPrimeValue += trajectoryWeights.get(i) * sum;
		}
		
		return lPrimeValue;
	}
	
	// Jerry
	/**
	 * Computes theta_{t+1} = theta_t + alpha * G(L)
	 * Computes new feature weights by taking a small step in the positive step of the gradient
	 * @param featureWeights
	 * @param trajectoryWeights
	 * @param policy
	 * @param episodes
	 * @return New feature weights
	 */
	protected static ApprenticeshipLearning.FeatureWeights computeNewFeatureWeightsViaGradientAscent(
			ApprenticeshipLearning.FeatureWeights featureWeights, List<Double> trajectoryWeights, 
			BoltzmannPolicySum bPolicySum,  List<EpisodeAnalysis> episodes, 
			State startState, Domain domain,
			StateHashFactory hashFactory,double alpha ) {
		int kIterationNum = 1;
		double lPrimeValue = 0.0;
		double[] weights = featureWeights.getWeights();
		double[] dWeights = new double[weights.length];
		// weights and rewards will not change for one iteration
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
		
		return new ApprenticeshipLearning.FeatureWeights(weights, 1.0);
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
	
	/**
	 * A single step in the EM algorithm
	 * @param request
	 * @param episodes
	 * @param featureWeightClusters
	 * @param clusterPriorProbabilities
	 * @param policies
	 */
	protected static void stepEM(MLIRLRequest request, List<EpisodeAnalysis> episodes, List<double[]> featureWeightClusters,
			double[] clusterPriorProbabilities, List<Policy> policies) {
		
		// E Step
		// Compute z_ij = prod pi * rho / Z
		List<double[]> trajectoryInClustersProbabilities = computeTrajectoryInClusterProbabilities(
				policies, episodes, clusterPriorProbabilities);
		
		// M Step
		// Compute rho_l = sum z_il / N
		clusterPriorProbabilities = computePriors(trajectoryInClustersProbabilities);
		
		// Compute theta_l via MLIRL and generate appropriate policies
		policies = computeClusterPolicies(request, trajectoryInClustersProbabilities);
		
		//policies = this.computePolicies(request.getPlanner(), featureWeightClusters);
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
	protected static List<double[]> computeTrajectoryInClusterProbabilities(List<Policy> policies, List<EpisodeAnalysis> trajectories, double[] clusterPriorProbabilities) {
		List<double[]> trajectoryInClusterProbabilities = 
				new ArrayList<double[]>(trajectories.size());
		
		int numberClusters = policies.size();
		EpisodeAnalysis episode;
		Policy policy;
		double prior;
		double sum;
		
		for (int i = 0; i < trajectories.size(); i++) {
			double[] probabilities = new double[numberClusters];
			sum = 0;
			for (int j = 0; j < numberClusters; j++) {
				episode = trajectories.get(i);
				policy = policies.get(j);
				prior = clusterPriorProbabilities[j];
				probabilities[j] = computeTrajectoryInClusterProbability(episode, policy, prior);
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
	protected static double computeTrajectoryInClusterProbability(EpisodeAnalysis trajectory, Policy policy, double prior) {
		double product = prior;
		List<GroundedAction> actions = trajectory.actionSequence;
		List<State> states = trajectory.stateSequence;
		int trajectoryLength = Math.min(actions.size(), states.size());
		
		State state;
		GroundedAction action;
		for (int i = 0; i < trajectoryLength; i++) {
			state = states.get(i);
			action = actions.get(i);
			product *= policy.getProbOfAction(state, action);
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
	 * @param trajectoryInClustersProbabilities
	 * @return The policies generated for each cluster
	 */
	protected static List<Policy> computeClusterPolicies(MLIRLRequest request, List<double[]> trajectoryInClustersProbabilities) {
		int numberClusters = trajectoryInClustersProbabilities.size();
		MLIRLRequest newRequest = new MLIRLRequest(request);
		List<Policy> policies = new ArrayList<Policy>(numberClusters);
		Policy policy;
		double[] trajectoryWeights;
		
		for (int i = 0; i < numberClusters; i++) {
			trajectoryWeights = trajectoryInClustersProbabilities.get(i);
			newRequest.setTrajectoryWeights(trajectoryWeights);
			policy = runMLIRL(newRequest);
			policies.add(policy);
		}
		
		return policies;
	}
}
