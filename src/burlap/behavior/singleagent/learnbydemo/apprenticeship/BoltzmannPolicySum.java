package burlap.behavior.singleagent.learnbydemo.apprenticeship;

import java.util.HashMap;
import java.util.List;

import burlap.behavior.singleagent.QValue;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TransitionProbability;
import burlap.oomdp.singleagent.GroundedAction;

public class BoltzmannPolicySum {
	private HashMap<StateHashTuple, Double> zValues ;
	private HashMap<StateHashTuple, Double> zPrimeValues;
	private QComputablePlanner planner;
	private double beta;
	private double gamma;
	private ValueFunctionValues valueFunction;
	private StateHashFactory stateHashFactory;
	public BoltzmannPolicySum(QComputablePlanner planner, double beta) {
		this.planner = planner;
		this.beta = beta;
		this.stateHashFactory = ((OOMDPPlanner)planner).getHashingFactory();
	}
	
	private void setValueFunction(ValueFunctionValues valueFunction) {
		this.valueFunction = valueFunction;
	}
	
	public double getZ(State state) {
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		Double v = this.zValues.get(hash);
		if (v != null) {
			return v.doubleValue();
		}
		return this.computeZ(hash, this.planner.getQs(state));
	}
	
	private double computeZ(StateHashTuple hash, List<QValue> qValues) {
		double Z = 0.0;
		for (QValue q : qValues) {
			Z += Math.exp(this.beta * q.q);
		}
		this.zValues.put(hash, Z);
		return Z;
	}

	public double getZPrime(State state, int feature) {
		StateHashTuple hash = this.stateHashFactory.hashState(state);
		Double value = this.zPrimeValues.get(hash);
		if (value != null) {
			return value.doubleValue();
		}
		
		List<QValue> qValues = this.planner.getQs(state);
		return this.computeZPrime(state, hash, qValues, feature);
		
	}
	
	public double computeZPrime(State state, StateHashTuple hash, List<QValue> qValues, int feature) {
		double ZPrime = 0.0;
		
		for (QValue q : qValues) {
			GroundedAction ga = (GroundedAction)q.a;
			List<TransitionProbability> transitions = ga.action.getTransitions(state, ga.params);
			double innerSum = 0.0;
			for (TransitionProbability tp : transitions) {
				innerSum += tp.p * this.valueFunction.getValuePrime(tp.s);
			}
			ZPrime += Math.exp(this.beta * q.q) * (feature + this.gamma * innerSum);
		}
		ZPrime *= this.beta;
		this.zPrimeValues.put(hash, ZPrime);
		return ZPrime;
	}
	
}
