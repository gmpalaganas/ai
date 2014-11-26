package ai.ann;
import java.lang.invoke.SerializedLambda;

/**
*	Functional Interface for the Activation Function Lambda
*/


@FunctionalInterface
public interface ActvFunct extends java.io.Serializable{
	public double activate(double net);
}
