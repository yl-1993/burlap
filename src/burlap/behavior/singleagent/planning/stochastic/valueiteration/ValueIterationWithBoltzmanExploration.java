package burlap.behavior.singleagent.planning.stochastic.valueiteration;

import java.util.ArrayList;
import java.util.List;

import burlap.behavior.singleagent.options.Option;
import burlap.behavior.singleagent.planning.ActionTransitions;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.TransitionProbability;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class ValueIterationWithBoltzmanExploration extends ValueIteration {

	public static final double DEFAULT_BETA = 0.5;
	protected double beta;
	
	public ValueIterationWithBoltzmanExploration(Domain domain,
			RewardFunction rf, TerminalFunction tf, double gamma,
			StateHashFactory hashingFactory, double maxDelta, int maxIterations) {
		super(domain, rf, tf, gamma, hashingFactory, maxDelta, maxIterations);
		this.beta = DEFAULT_BETA;
	}
	
	public ValueIterationWithBoltzmanExploration(Domain domain,
			RewardFunction rf, TerminalFunction tf, double gamma,
			StateHashFactory hashingFactory, double maxDelta, int maxIterations, double beta) {
		super(domain, rf, tf, gamma, hashingFactory, maxDelta, maxIterations);
		this.beta = beta;
	}
	
	@Override
	protected double performBellmanUpdateOn(StateHashTuple sh){
		if(this.tf.isTerminal(sh.s)){
			//terminal states always have a state value of 0
			valueFunction.put(sh, 0.);
			return 0.;
		}
		
		
		double maxQ = Double.NEGATIVE_INFINITY;
		
		if(this.useCachedTransitions){
		
			List<ActionTransitions> transitions = this.getActionsTransitions(sh);
			for(ActionTransitions at : transitions){
				double q = this.computeQ(sh.s, at);
				if(q > maxQ){
					maxQ = q;
				}
			}
			
		}
		else{
			
			List <GroundedAction> gas = sh.s.getAllGroundedActionsFor(this.actions);
			for(GroundedAction ga : gas){
				double q = this.computeQ(sh, ga);
				if(q > maxQ){
					maxQ = q;
				}
			}
			
		}
		
		valueFunction.put(sh, maxQ);
		
		return maxQ;
	}
	
	protected double blendQValuesViaBoltzmann(StateHashTuple sh, List <TransitionProbability> tps) {
		double q = 0.0;
		
		double sum = 0.0;
		List<Double> qValues = new ArrayList<Double>(tps.size());
		
		for (TransitionProbability tp : tps) {
			double value = this.value(tp.s);
			double exponentiatedQ = Math.exp(this.beta * value);
			qValues.add(exponentiatedQ);
			sum += exponentiatedQ;
		}
		
		for (Double qValue : qValues) {
			q += qValue / sum;
		}
		
		return q;
	}
	
	@Override
	protected double computeQ(StateHashTuple sh, GroundedAction ga) {
		double q = 0.0;
		
		if(ga.action instanceof Option){
			
			Option o = (Option)ga.action;
			double expectedR = o.getExpectedRewards(sh.s, ga.params);
			q += expectedR;
			
			List <TransitionProbability> tps = o.getTransitions(sh.s, ga.params);
			for(TransitionProbability tp : tps){
				double vp = this.blendQValuesViaBoltzmann(sh, tps);
				
				//note that for options, tp.p will be the *discounted* probability of transition to s',
				//so there is no need for a discount factor to be included
				q += tp.p * vp; 
			}
			
		}
		else{
			
			List <TransitionProbability> tps = ga.action.getTransitions(sh.s, ga.params);
			for(TransitionProbability tp : tps){
				double vp = this.blendQValuesViaBoltzmann(sh, tps);
				double discount = this.gamma;
				double r = rf.reward(sh.s, ga, tp.s);
				
				q += tp.p * (r + (discount * vp));
			}
			
		}
		
		return q;
	}

}
