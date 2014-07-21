package burlap.behavior.singleagent.learnbydemo.apprenticeship;

/**
 * Class of feature weights which contain the weight values and the associated score given to them
 * @author Stephen Brawner
 *
 */
public class FeatureWeights {
	private double[] weights;

	public FeatureWeights(double[] weights) {
		this.weights = weights.clone();
	}

	public FeatureWeights(FeatureWeights featureWeights) {
		this.weights = featureWeights.getWeights();
	}

	public double[] getWeights() {
		return this.weights.clone();
	}
}
