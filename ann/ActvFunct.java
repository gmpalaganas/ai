package ai.ann;
import java.lang.invoke.SerializedLambda;

@FunctionalInterface
public interface ActvFunct extends java.io.Serializable{
	public double activate(double net);
}
