package burlap.behavior.singleagent.learnbydemo.apprenticeship.requests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;

public class MLIRLRequest {

	/**
	 * The domain in which IRL is to be performed
	 */
	protected Domain 								domain;
	
	/**
	 * The planning algorithm used to compute the policy for a given reward function
	 */
	protected OOMDPPlanner 							planner;
	
	/**
	 * The state feature generator that turns a state into a feature vector on which the reward function is assumed to be modeled
	 */
	protected StateToFeatureVectorGenerator 		featureGenerator;
	
	
	/**
	 * Length of feature vector
	 */
	protected int									featureVectorLength;
	
	/**
	 * The input trajectories/episodes that are to be modeled.
	 */
	protected List<EpisodeAnalysis> 				trajectories;
	
	/**
	 * The weight given to each trajectory/episode;
	 */
	protected List<Double>							trajectoryWeights;
	
	
	/**
	 * The number of clusters used in EM
	 */
	protected int									numberClusters;
	
	/**
	 * The initial state generator that models the initial states from which the expert trajectories were drawn
	 */
	protected StateGenerator 						startStateGenerator;
	
	/**
	 * The discount factor of the problem
	 */
	protected double 								gamma;
	
	/**
	 * The maximum feature score to cause termination of Apprenticeship learning
	 */
	protected double 								epsilon;
	
	/**
	 * The temperature value used for Boltzmann exploration
	 */
	protected double								beta;
	
	/**
	 * The maximum number of iterations of apprenticeship learning
	 */
	protected int 									maxIterations;
	
	public static final double 			DEFAULT_GAMMA = 0.99;
	public static final double 			DEFAULT_EPSILON = 0.01;
	public static final double			DEFAULT_BETA = 0.5;
	public static final int 			DEFAULT_MAXITERATIONS = 100;
	public static final int 			DEFAULT_POLICYCOUNT = 5;	

	public MLIRLRequest() {
		this.initDefaults();
	}

	public MLIRLRequest(Domain domain, OOMDPPlanner planner, StateToFeatureVectorGenerator featureGenerator, List<EpisodeAnalysis> expertEpisodes, StateGenerator startStateGenerator) {
		this.initDefaults();
		this.setDomain(domain);
		this.setPlanner(planner);
		this.setFeatureGenerator(featureGenerator);
		this.setExpertEpisodes(expertEpisodes);
		this.setStartStateGenerator(startStateGenerator);
	}

	public MLIRLRequest(MLIRLRequest request) {
		this.initDefaults();
		this.setDomain(request.getDomain());
		this.setPlanner(request.getPlanner());
		this.setFeatureGenerator(request.getFeatureGenerator());
		this.setExpertEpisodes(request.getExpertEpisodes());
		this.setStartStateGenerator(request.getStartStateGenerator());
	}

	private void initDefaults() {
		this.gamma = MLIRLRequest.DEFAULT_GAMMA;
		this.epsilon = MLIRLRequest.DEFAULT_EPSILON;
		this.maxIterations = MLIRLRequest.DEFAULT_MAXITERATIONS;
		this.beta = MLIRLRequest.DEFAULT_BETA;
	}

	public boolean isValid() {
		if (this.domain == null) {
			return false;
		}
		if (this.planner == null) {
			return false;
		}
		if (this.featureGenerator == null) {
			return false;
		}
		if (this.trajectories.size() == 0) {
			return false;
		}
		if (this.startStateGenerator == null) {
			return false;
		}
		if (this.gamma > 1 || this.gamma < 0 || Double.isNaN(this.gamma)) {
			return false;
		}
		if (this.epsilon < 0 || Double.isNaN(this.epsilon)) {
			return false;
		}
		if (this.maxIterations <= 0) {
			return false;
		}
		return true;
	}

	public void setDomain(Domain d) {
		this.domain = d;
	}


	public void setPlanner(OOMDPPlanner p) {
		this.planner = p;
	}

	public void setFeatureGenerator(StateToFeatureVectorGenerator stateFeaturesGenerator) {
		this.featureGenerator = stateFeaturesGenerator;
	}
	
	public void setFeatureVectorLength(int length) {
		this.featureVectorLength = length;
	}

	public void setExpertEpisodes(List<EpisodeAnalysis> episodeList) {
		this.trajectories = new ArrayList<EpisodeAnalysis>(episodeList);
	}
	
	public void setTrajectoryWeights(double[] trajectoryWeights) {
		this.trajectoryWeights = new ArrayList<Double>(trajectoryWeights.length);
		for (double d : trajectoryWeights) {
			this.trajectoryWeights.add(d);
		}
		Double[] weights = ArrayUtils.toObject(trajectoryWeights);
		this.setTrajectoryWeights(weights);
	}
	
	public void setTrajectoryWeights(Double[] trajectoryWeights) {
		this.trajectoryWeights = Arrays.asList(trajectoryWeights);
	}
	
	public void setNumberClusters(int numberClusters) {
		this.numberClusters = numberClusters;
	}

	public void setStartStateGenerator(StateGenerator startStateGenerator) { this.startStateGenerator = startStateGenerator;}

	public void setGamma(double gamma) { this.gamma = gamma;}

	public void setBeta(double beta) {this.beta = beta;}
	
	public void setEpsilon(double epsilon) {this.epsilon = epsilon;}

	public void setMaxIterations(int maxIterations) {this.maxIterations = maxIterations;}

	public Domain getDomain() {return this.domain;}

	public OOMDPPlanner getPlanner() {return this.planner;}
	
	public TerminalFunction getTerminalFunction() {return this.planner.getTF();}
	
	public StateHashFactory getStateHashFactory() {return this.planner.getHashingFactory();}

	public StateToFeatureVectorGenerator getFeatureGenerator() {return this.featureGenerator;}	
	
	public int getFeatureVectorLength() {return this.featureVectorLength;}

	public List<EpisodeAnalysis> getExpertEpisodes() { return new ArrayList<EpisodeAnalysis>(this.trajectories);}

	public List<Double> getEpisodeWeights() {return new ArrayList<Double>(this.trajectoryWeights);}
	
	public int getNumberClusters() {return this.numberClusters;}
	
	public StateGenerator getStartStateGenerator() {return this.startStateGenerator;}

	public double getGamma() {return this.gamma;}

	public double getBeta() {return this.beta;}
	
	public double getEpsilon() {return this.epsilon;}

	public int getMaxIterations() {return this.maxIterations;}
}
