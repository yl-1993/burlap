package burlap.behavior.singleagent.learnbydemo.apprenticeship;

import java.util.HashMap;
import java.util.List;
import burlap.behavior.singleagent.QValue;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TransitionProbability;
import burlap.oomdp.singleagent.GroundedAction;

public class BoltzmannPolicySum {
	private HashMap<StateHashTuple, Double> rewardValues;
	private HashMap<StateActionTuple, Double> qValues;
	private HashMap<QFeatureTuple, Double> qPrimeValues;
	private HashMap<StateHashTuple, Double> zValues;
	private HashMap<HashFeatureTuple, Double> zPrimeValues;
	private HashMap<StateActionTuple, Double> piValues;
	private HashMap<QFeatureTuple, Double> piPrimeValues;
	private HashMap<StateHashTuple, Double> vValues;
	private HashMap<StateHashTuple, Double> vLastValues;
	private HashMap<HashFeatureTuple, Double> vPrimeValues;
	private HashMap<HashFeatureTuple, Double> vLastPrimeValues; 
	private QComputablePlanner planner;
	private StateToFeatureVectorGenerator featureGenerator;
	private FeatureWeights featureWeights;
	private double beta;
	private double gamma;
	private StateHashFactory stateHashFactory;
	
	public class StateActionTuple{
		public State s;
		public GroundedAction a;
		
		public StateActionTuple(State s, GroundedAction a){
			this.s = s;
			this.a = a;
		}
		
		public StateActionTuple(StateActionTuple sa){
			this.s = sa.s;
			this.a = sa.a;
		}
		
		@Override
		public boolean equals(Object other){
			if(this == other){
				return true;
			}
			if(!(other instanceof StateActionTuple)){
				return false;
			}
			StateActionTuple o = (StateActionTuple)other;
			return (this.s.equals(o.s) && this.a.equals(o.a));
		}
		
		@Override
		public int hashCode(){
			// hashCode return the memory address
			int res = this.s.toString().hashCode() + this.a.toString().hashCode();
			return res;
		}
	}
	
	public class QFeatureTuple{
		public StateActionTuple sa;
		public int feature;
		
		public QFeatureTuple(StateActionTuple sa, int feature){
			this.sa = sa;
			this.feature = feature;
		}
		
		public QFeatureTuple(QFeatureTuple qf){
			this.sa = qf.sa;
			this.feature = qf.feature;
		}
		
		
		@Override
		public boolean equals(Object other){
			if(this == other){
				return true;
			}
			if(!(other instanceof QFeatureTuple)){
				return false;
			}
			QFeatureTuple o = (QFeatureTuple)other;
			return (this.sa.equals(o.sa) && this.feature == o.feature);	
		}
		
		@Override
		public int hashCode(){
			int res = this.feature + this.sa.hashCode();
			return res;
		}
	}
	
	public class HashFeatureTuple{
		public StateHashTuple hash;
		public int feature;
		
		public HashFeatureTuple(StateHashTuple hash, int feature){
			this.hash = hash;
			this.feature = feature;
		}
		
		public HashFeatureTuple(HashFeatureTuple hf){
			this.hash = hf.hash;
			this.feature = hf.feature;
		}
		
		
		@Override
		public boolean equals(Object other){
			if(this == other){
				return true;
			}
			if(!(other instanceof HashFeatureTuple)){
				return false;
			}
			HashFeatureTuple o = (HashFeatureTuple)other;
			return (this.hash.equals(o.hash) && this.feature == o.feature);	
		}
		
		@Override
		public int hashCode(){
			int res = this.feature + this.hash.hashCode();
			return res;
		}
	}
	
	public BoltzmannPolicySum(QComputablePlanner planner, 
			StateToFeatureVectorGenerator featureGenerator, 
			FeatureWeights featureWeights, 
			double beta, double gamma) {
		this.planner = planner;
		this.beta = beta;
		this.gamma = gamma;
		this.stateHashFactory = ((OOMDPPlanner)planner).getHashingFactory();
		this.rewardValues = new HashMap<>();
		this.qValues = new HashMap<>();
		this.qPrimeValues = new HashMap<>();
		this.zValues = new HashMap<>();
		this.zPrimeValues = new HashMap<>();
		this.piValues = new HashMap<>();
		this.piPrimeValues = new HashMap<>();
		this.vValues = new HashMap<>();
		this.vLastValues = new HashMap<>();
		this.vPrimeValues = new HashMap<>();
		this.vLastPrimeValues = new HashMap<>();
		this.featureGenerator = featureGenerator;
		this.featureWeights = featureWeights;
	}
	

	public double getQ(QValue qValue, State state){
		StateActionTuple saTuple = new StateActionTuple(state, (GroundedAction)qValue.a);
		Double value = this.qValues.get(saTuple);
		if (value != null) {
			return value.doubleValue();
		}
		return computeQ(qValue, state);
	}
	
	public double computeQ(QValue q, State state){
		GroundedAction ga = (GroundedAction)q.a;
		List<TransitionProbability> transitions = ga.action.getTransitions(state, ga.params);
		double innerSum = 0.0;
		for (TransitionProbability tp : transitions) {
			//innerSum += tp.p * this.valueFunction.getValuePrime(tp.s);
			//innerSum += tp.p * this.getValuePrime(tp.s, feature);
			StateHashTuple nextHash = this.stateHashFactory.hashState(tp.s);
			innerSum += (tp.p * this.vLastValues.get(nextHash));
		}
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		double qValue = this.rewardValues.get(hash) + this.gamma*innerSum;	
		StateActionTuple saTuple = new StateActionTuple(state, ga);
		this.qValues.put(saTuple, qValue);
		return qValue;
	}
	
	/**
	 * 
	 * @param q
	 * @param state
	 * @param feature
	 * @return
	 */
	public double getQPrime(QValue q, State state, int feature){
		StateActionTuple saTuple = new StateActionTuple(state,(GroundedAction)q.a);
		QFeatureTuple qfTuple = new QFeatureTuple(saTuple, feature);
		Double value = this.qPrimeValues.get(qfTuple);
		if (value != null) {
			return value.doubleValue();
		}
		return computeQPrime(q, state, feature);
	}
	
	public double computeQPrime(QValue q, State state, int feature){
		GroundedAction ga = (GroundedAction)q.a;
		List<TransitionProbability> transitions = ga.action.getTransitions(state, ga.params);
		double innerSum = 0.0;
		for (TransitionProbability tp : transitions) {
			//innerSum += tp.p * this.valueFunction.getValuePrime(tp.s);
			//innerSum += tp.p * this.getValuePrime(tp.s, feature);
			StateHashTuple nextHash = this.stateHashFactory.hashState(tp.s);
			HashFeatureTuple hfTuple = new HashFeatureTuple(nextHash, feature);
			innerSum += (tp.p * this.vLastPrimeValues.get(hfTuple));
		}
		double[] fv = featureGenerator.generateFeatureVectorFrom(state);
		double qPrime = fv[feature] + this.gamma * innerSum;
		StateActionTuple saTuple = new StateActionTuple(state, ga);
		QFeatureTuple qfTuple = new QFeatureTuple(saTuple, feature);
		this.qPrimeValues.put(qfTuple, qPrime);
		return qPrime;
	}
	
	public double getZ(State state) {
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		Double value = this.zValues.get(hash);
		if (value != null) {
			return value.doubleValue();
		}
		
		List<QValue> qValues = this.planner.getQs(state);
		return this.computeZ(state, hash, qValues);
	}
	
	public double computeZ(State state,StateHashTuple hash, List<QValue> qValues) {
		double Z = 0.0;
		for (QValue qValue : qValues) {
			Z += Math.exp(this.beta * this.getQ(qValue, state));
		}
		this.zValues.put(hash, Z);
		return Z;
	}


	
	public double getZPrime(State state, int feature) {
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		HashFeatureTuple hfTuple = new HashFeatureTuple(hash,feature);
		Double value = this.zPrimeValues.get(hfTuple);
		if (value != null) {
			return value.doubleValue();
		}
		
		List<QValue> qValues = this.planner.getQs(state);
		return this.computeZPrime(state, hash, qValues, feature);
		
	}
	
	public double computeZPrime(State state, StateHashTuple hash, List<QValue> qValues, int feature) {
		double ZPrime = 0.0;
		
		for (QValue qValue : qValues) {
			ZPrime += (Math.exp(this.beta * this.getQ(qValue, state))
					* this.getQPrime(qValue, state, feature));
		}
		ZPrime *= this.beta;
		HashFeatureTuple hfTuple = new HashFeatureTuple(hash,feature);
		this.zPrimeValues.put(hfTuple, ZPrime);
		return ZPrime;
	}
	
	public double getPi(State state, QValue qValue){
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		StateActionTuple saTuple = new StateActionTuple(state, (GroundedAction)qValue.a);
		Double value = this.piValues.get(saTuple);
		if (value != null) {
			return value.doubleValue();
		}	
		return computePi(state, qValue);
	}
	
	public double computePi(State state, QValue qValue){
		double Pi = Math.exp(this.beta * this.getQ(qValue, state)) / this.getZ(state);
		StateActionTuple saTuple = new StateActionTuple(state, (GroundedAction)qValue.a);
		this.piValues.put(saTuple, Pi);
		return Pi;
	}
	
	public double getPiPrime(State state, QValue qValue, int feature){
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		StateActionTuple saTuple = new StateActionTuple(state, (GroundedAction)qValue.a);
		QFeatureTuple qfTuple = new QFeatureTuple(saTuple, feature);
		Double value = this.piPrimeValues.get(qfTuple);
		if (value != null) {
			return value.doubleValue();
		}		
		return this.computePiPrime(state, hash, qValue, feature);
	}
	
	public double computePiPrime(State state, StateHashTuple hash, QValue qValue, int feature){
		double piPrime = 0.0;
		double z = this.getZ(state);
		double zPrime = this.getZPrime(state, feature);
		double q = this.getQ(qValue, state);
		double qPrime = this.getQPrime(qValue, state, feature);
		//
		double den = z*z;
		double num = this.beta * z * qPrime / den - zPrime / den;
		
		//piPrime = this.beta*getPi(state, qValue)*qPrime - getPi(state, qValue)*zPrime/z;
		piPrime = num*Math.exp(this.beta * q);
		
		StateActionTuple saTuple = new StateActionTuple(state, (GroundedAction)qValue.a);
		QFeatureTuple qfTuple = new QFeatureTuple(saTuple,feature);
		this.piPrimeValues.put(qfTuple, piPrime);
		return piPrime;	
	}
	
	public double getValue(State state){
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		Double value = this.vValues.get(hash);
		if (value != null) {
			return value.doubleValue();
		}	
		List<QValue> qValues = this.planner.getQs(state);
		return computeValue(state, hash, qValues);		
	}
	
	public double computeValue(State state, StateHashTuple hash, List<QValue> qValues){
		double value = 0.0;
		// value = sum_a pi_i(s,a)*Q_i(s,a)
		for (QValue qValue : qValues) {
			value += (this.getPi(state, qValue) * this.getQ(qValue, state));
		}
		
		this.vValues.put(hash, value);
		return value;
	}
	
	public double getValuePrime(State state, int feature){
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		HashFeatureTuple hfTuple = new HashFeatureTuple(hash, feature);
		Double value = this.vPrimeValues.get(hfTuple);
		if (value != null) {
			return value.doubleValue();
		}	
		List<QValue> qValues = this.planner.getQs(state);
		return computeValuePrime(state, hash, qValues, feature);
	}
	
	public double computeValuePrime(State state, StateHashTuple hash, List<QValue> qValues, int feature){
		double vPrime = 0.0;
		//vPrime = sum_a (Q_i(s,a)*d(pi_i(s,a)/d w_j) + pi_i(s,a)*d(Q_i(s,a)/d w_j))
		for (QValue qValue : qValues) {
			vPrime += ((this.getQ(qValue, state) * this.getPiPrime(state, qValue, feature) 
					+ this.getPi(state, qValue) * this.getQPrime(qValue, state, feature)));
		}
		HashFeatureTuple hfTuple = new HashFeatureTuple(hash, feature);
		this.vPrimeValues.put(hfTuple, vPrime);
		return vPrime;
	}
	
	public void updateLastValue(State state){
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		if (this.vLastValues.get(hash) == null) {
			// initialize vValue
			double sum = 0.0;
			double[] fvWeights = this.featureWeights.getWeights();
			double[] fv = this.featureGenerator.generateFeatureVectorFrom(state);
			for (int i = 0; i < fv.length; i++) {
				sum += (fv[i] * fvWeights[i]);
			}
			this.vLastValues.put(hash, sum);
		}
		else{
			Double value = this.vValues.get(hash);
			if(value != null){
				this.vLastValues.put(hash, value);
				this.vValues.put(hash, null);
			}
			else {
				return;
			}
		}
	}
	
	public void updateLastValuePrime(State state, int feature){
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		HashFeatureTuple hfTuple = new HashFeatureTuple(hash, feature);
		if (this.vLastPrimeValues.get(hfTuple) == null) {
			// initialize vPrime
			double[] fv = this.featureGenerator.generateFeatureVectorFrom(state);
			this.vLastPrimeValues.put(hfTuple, fv[feature]);
		}
		else {
			Double value = this.vPrimeValues.get(hfTuple);
			if(value != null){
				this.vLastPrimeValues.put(hfTuple, value);
				this.vPrimeValues.put(hfTuple, null); // clear V'_{t} to compute V'_{t+1}
			}
			else {
				return;
			}
		}
	}
	
	public void updateRewardValues(State state){
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		double reward = 0.0;
		double[] fv = this.featureGenerator.generateFeatureVectorFrom(state);
		double[] fvWeights = this.featureWeights.getWeights();
		for (int i = 0; i < fv.length; i++) {
			reward += (fv[i]*fvWeights[i]);
		}
		// update every time
		this.rewardValues.put(hash, reward);
	}
	
	public void updateFeatureWeights(double[] weights){
		this.featureWeights = new FeatureWeights(weights); // TODO: The second parameter is useless here 
	}
	
	public void clearAllValuesExceptV(){
		this.qValues.clear();
		this.qPrimeValues.clear();
		this.zValues.clear();
		this.zPrimeValues.clear();
		this.piValues.clear();
		this.piPrimeValues.clear();
	}
}
