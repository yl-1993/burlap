package burlap.behavior.singleagent.learnbydemo.apprenticeship;

public class FeatureWeightsWithScores extends FeatureWeights{
	
	private double score;
	
	public FeatureWeightsWithScores(double[] weights){
		super(weights);
		this.score = 0;
	}
	
	public FeatureWeightsWithScores(double[]weights, double score) {
		super(weights);
		this.score = score;
	}
	
	public FeatureWeightsWithScores(FeatureWeightsWithScores featureWeights) {
		super(featureWeights.getWeights());
		this.score = featureWeights.getScore();
	}
	
	public Double getScore() {
		return this.score;
	}
}
