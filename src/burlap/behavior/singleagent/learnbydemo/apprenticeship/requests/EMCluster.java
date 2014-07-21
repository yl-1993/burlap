package burlap.behavior.singleagent.learnbydemo.apprenticeship.requests;

public abstract class EMCluster {
	/**
	 * The number of clusters used in EM
	 */
	protected int									numberClusters;
	
	public static final int 			DEFAULT_NUMBERCLUSTERS = 1;
	
	public EMCluster(){
		this.numberClusters = EMCluster.DEFAULT_NUMBERCLUSTERS;
	}
	
	public EMCluster(int numberClusters) {
		this.numberClusters = numberClusters;
	}
	
	public void setNumberClusters(int numberClusters) {
		this.numberClusters = numberClusters;
	}
	
	public int getNumberClusters() {
		return this.numberClusters;
	}
	
}
