package burlap.behavior.singleagent.learnbydemo.apprenticeship.requests;

import java.util.List;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.core.Domain;

public class EMMLIRLRequest extends MLIRLRequest {
	/**
	 * The number of clusters used in EM
	 */
	protected int									numberClusters;
	
	public static final int 			DEFAULT_NUMBERCLUSTERS = 1;
	
	public EMMLIRLRequest(){
		this.numberClusters = EMMLIRLRequest.DEFAULT_NUMBERCLUSTERS;
	}
	
	public EMMLIRLRequest(int numberClusters) {
		this.numberClusters = numberClusters;
	}
	
	public EMMLIRLRequest(Domain domain, OOMDPPlanner planner,
			StateToFeatureVectorGenerator featureGenerator, 
			List<EpisodeAnalysis> expertEpisodes, 
			StateGenerator startStateGenerator,
			int numberClusters) {
		super();
		this.setDomain(domain);
		this.setPlanner(planner);
		this.setFeatureGenerator(featureGenerator);
		this.setExpertEpisodes(expertEpisodes);
		this.setStartStateGenerator(startStateGenerator);
		this.setFeatureVectorLength(domain.getPropFunctions().size());
		this.setNumberClusters(numberClusters);
	}
	
	public EMMLIRLRequest(EMMLIRLRequest request){
		super();
		this.setDomain(request.getDomain());
		this.setPlanner(request.getPlanner());
		this.setFeatureGenerator(request.getFeatureGenerator());
		this.setExpertEpisodes(request.getExpertEpisodes());
		this.setStartStateGenerator(request.getStartStateGenerator());
		this.setFeatureVectorLength(request.getDomain().getPropFunctions().size());
	}
	
	public void setNumberClusters(int numberClusters) {
		this.numberClusters = numberClusters;
	}
	
	public int getNumberClusters() {
		return this.numberClusters;
	}
	
}
