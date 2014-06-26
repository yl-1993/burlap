package burlap.behavior.singleagent.learnbydemo.apprenticeship;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import org.apache.commons.lang3.ArrayUtils;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
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
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class MultipleIntentionIRL {


	public Policy runMLIRL(MLIRLRequest request) {
		
		// Get all required items from the request object
		Domain domain = request.getDomain();
		StateToFeatureVectorGenerator featureGenerator = request.getFeatureGenerator();
		TerminalFunction terminalFunction = request.getTerminalFunction();
		Double gamma = request.getGamma();
		StateHashFactory hashFactory = request.getStateHashFactory();
		double beta = request.getBeta();
		List<Double> trajectoryWeights = request.getEpisodeWeights();
		List<EpisodeAnalysis> trajectories = request.getExpertEpisodes();
		OOMDPPlanner planner = request.getPlanner();
		int numIterations = request.getMaxIterations();
		int featureVectorLength = request.getFeatureVectorLength();

		// Initialize choose random set of reward weights
		ApprenticeshipLearning.FeatureWeights featureWeights = 
				this.getRandomRewardWeights(featureVectorLength);
		
		// Generate reward function from random weights
		RewardFunction rewardFunction = 
				ApprenticeshipLearning.generateRewardFunction(featureGenerator, featureWeights);
		
		Policy policy = null;
		// for t = 1 to M do
		for (int i = 0; i < numIterations; i++) {
			// Need to reset planner every time to accommodate new reward function
			planner.resetPlannerResults();
			planner.plannerInit(domain, rewardFunction, terminalFunction, gamma , hashFactory);
			
			// Generate random start state
			State startState = request.getStartStateGenerator().generateState();
			
			// From feature weights, compute Q_theta_t, and generate a new policy from the Q function
			policy = computePolicyFromRewardWeights(planner, featureWeights, featureGenerator, startState, beta);
			
			// TODO set a realistic alpha
			double alpha = 1.0;
			// Compute new feature weights through gradient ascent, theta_{t+1} = theta_t + alpha * G(L)
			featureWeights = this.computeNewFeatureWeightsViaGradientAscent(featureWeights, trajectoryWeights, policy, trajectories, alpha);
			
			// Compute log likelihood of data (L)
			double logLikelihood = this.computeLogLikelihoodOfTrajectories(trajectoryWeights, policy, trajectories);
		}
		
		return policy;
	}


	public void runEM(MLIRLRequest request) {
		
		// Get all required items for EM
		int numberClusters = request.getNumberClusters();
		List<EpisodeAnalysis> episodes = request.getExpertEpisodes();
		int featureWeightLength = request.getFeatureVectorLength();
		int numberIterations = request.getMaxIterations();
		
		List<double[]> featureWeightClusters =
				new ArrayList<double[]>(numberClusters);
		
		// Randomly initialize the prior probabilities of each cluster
		ApprenticeshipLearning.FeatureWeights clusterPriors = 
				this.getRandomRewardWeights(numberClusters);
		double[] clusterPriorProbabilities = clusterPriors.getWeights();
		
		// Randomly initialize the feature weights of each cluster
		for (int i = 0; i < numberClusters; i++) {
			ApprenticeshipLearning.FeatureWeights weights = 
					this.getRandomRewardWeights(featureWeightLength);
			featureWeightClusters.add(weights.getWeights());
		}
		
		// Generate policies from initial values
		List<Policy> policies = this.computeClusterPolicies(request, featureWeightClusters);
		
		// repeat until target number of iterations completed
		for (int i = 0; i < numberIterations; i++) {
			stepEM(request, episodes, featureWeightClusters, clusterPriorProbabilities, policies);
		}
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
	protected Policy computePolicyFromRewardWeights(OOMDPPlanner planner, 
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
	protected double computeLogLikelihoodOfTrajectories(
			List<Double> trajectoryWeights, Policy policy, List<EpisodeAnalysis> trajectories) {
		
		double logLikelihood = 0.0;
		
		for (int i = 0; i < trajectories.size(); i++) {
			
			EpisodeAnalysis episode = trajectories.get(i);
			List<GroundedAction> actions = episode.actionSequence;
			List<State> states = episode.stateSequence;
			int trajectoryLength = Math.min(actions.size(), states.size());
			double sum = 0.0;
			
			for (int j = 0; j < trajectoryLength; j++) {
				GroundedAction ga = actions.get(j);
				State s = states.get(j);
				double probability = policy.getProbOfAction(s, ga); 
				sum += Math.log(probability);
			}
			
			logLikelihood += sum;
		}
		
		return logLikelihood;
	}
	
	/**
	 * Generates a random set of reward weights of given length
	 * @param length
	 * @return
	 */
	protected ApprenticeshipLearning.FeatureWeights getRandomRewardWeights(int length) {
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
	 * Computes theta_{t+1} = theta_t + alpha * G(L)
	 * Computes new feature weights by taking a small step in the positive step of the gradient
	 * @param featureWeights
	 * @param trajectoryWeights
	 * @param policy
	 * @param episodes
	 * @return New feature weights
	 */
	protected ApprenticeshipLearning.FeatureWeights computeNewFeatureWeightsViaGradientAscent(
			ApprenticeshipLearning.FeatureWeights featureWeights, List<Double> trajectoryWeights, 
			Policy policy, List<EpisodeAnalysis> episodes, double alpha ) {
		double[] weights = featureWeights.getWeights();
		
		for (int i = 0; i < weights.length; i++) {
			
			double[] dWeights = weights.clone();
			dWeights[i] += 0.001;
			//RewardFunction rewardFunction = 
			//		ApprenticeshipLearning.generateRewardFunction(featureGenerator, featureWeights);
			
			//double logLikelihood = 
		}
		
		return new ApprenticeshipLearning.FeatureWeights(weights, 1.0);
	}

	/**
	 * A single step in the EM algorithm
	 * @param request
	 * @param episodes
	 * @param featureWeightClusters
	 * @param clusterPriorProbabilities
	 * @param policies
	 */
	protected void stepEM(MLIRLRequest request, List<EpisodeAnalysis> episodes, List<double[]> featureWeightClusters,
			double[] clusterPriorProbabilities, List<Policy> policies) {
		
		// E Step
		// Compute z_ij = prod pi * rho / Z
		List<double[]> trajectoryInClustersProbabilities = this.computeTrajectoryInClusterProbabilities(
				policies, episodes, clusterPriorProbabilities);
		
		// M Step
		// Compute rho_l = sum z_il / N
		clusterPriorProbabilities = this.computePriors(trajectoryInClustersProbabilities);
		
		// Compute theta_l via MLIRL and generate appropriate policies
		policies = this.computeClusterPolicies(request, trajectoryInClustersProbabilities);
		
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
	protected List<double[]> computeTrajectoryInClusterProbabilities(List<Policy> policies, List<EpisodeAnalysis> trajectories, double[] clusterPriorProbabilities) {
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
				probabilities[j] = this.computeTrajectoryInClusterProbability(episode, policy, prior);
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
	protected double computeTrajectoryInClusterProbability(EpisodeAnalysis trajectory, Policy policy, double prior) {
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
	protected double[] computePriors(List<double[]> trajectoryInClustersProbabilities) {
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
	protected List<Policy> computeClusterPolicies(MLIRLRequest request, List<double[]> trajectoryInClustersProbabilities) {
		int numberClusters = trajectoryInClustersProbabilities.size();
		MLIRLRequest newRequest = new MLIRLRequest(request);
		List<Policy> policies = new ArrayList<Policy>(numberClusters);
		Policy policy;
		double[] trajectoryWeights;
		
		for (int i = 0; i < numberClusters; i++) {
			trajectoryWeights = trajectoryInClustersProbabilities.get(i);
			newRequest.setTrajectoryWeights(trajectoryWeights);
			policy = this.runMLIRL(newRequest);
			policies.add(policy);
		}
		
		return policies;
	}
}
