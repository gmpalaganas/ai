package ai.ann;
import java.util.*;


/**
*	Main body of an Artificial Neural Network
*/

public class Network implements java.io.Serializable{

	public List<Layer> layers; 
	public int n_bias;
	public ActvFunct f_act;

	public Network(int[] layers,int n_bias,ActvFunct f_act){

		this.f_act = f_act;
		this.n_bias = n_bias;
		this.layers = new ArrayList<Layer>();

		for(int i = 1; i < layers.length; i++){
			int n_in = layers[i-1];
			int n_nodes = layers[i];

			String name = ((i + 1) < layers.length)?"h" + i:"o";

			Layer l = new Layer(name,n_nodes,n_in,n_bias,f_act);

			this.layers.add(l);
		} 

	}


	public Network(int[] layers){
	
		this(layers,(ActvFunct)(net) -> (1/ (1 + (Math.pow(Math.E,-net)))));
	
	}

	public Network(int[] layers, ActvFunct f_act){
	
		this(layers,1,f_act);
	
	}

	public Network(int[] layers, int n_bias){
		
		this(layers,n_bias,(ActvFunct)(net) -> (1/ (1 + (Math.pow(Math.E,-net)))));
	
	}

	/**
	*	Computes the error for each <code>Perceptron</code>
	*	in the all <code>Layers</code> of the <code>Network</code>
	* 	then adjusts their weights
	*	@param target 	Array of target values
	*	@param lr 		Learning rate
	*	@param mtm		Momentum
	*/
	public void back_propagate(double[] target,double lr,double mtm){
		
		double[] o_errors = new double[target.length];
		int last = this.layers.size() - 1;
		Layer cur_layer = this.layers.get(last);

		for(int i = 0; i < target.length; i++)
			o_errors[i] = target[i] - cur_layer.perceptrons.get(i).output;
		
		cur_layer.back_propagate(o_errors,lr,mtm);
		last = last - 1;
		for(int i = 0; i < last + 1 ; i++){
			Layer prev_layer = cur_layer;
			cur_layer = this.layers.get(last - i);
			double[] h_errors = new double[cur_layer.get_n_nodes()];

			for(int j = 0; j < cur_layer.get_n_nodes(); j++){
				Perceptron cur_p = cur_layer.perceptrons.get(j);
				int p_index = j + this.n_bias;
					
				h_errors[j] = this.get_error(prev_layer.perceptrons,p_index,prev_layer.get_n_nodes());
			
			}

			cur_layer.back_propagate(h_errors,lr,mtm);

		}

	}

	/**
	*	Feeds the input into the <code>Network</code>
	*	then classifies the output
	*	@param inputs 	Array of inputs
	*	@return Array of classified outputs 
	*/
	public int[] classify(double[] inputs){

		double[] actual = this.feed_forward(inputs);
		int[] ret = new int[actual.length];

		for(int i = 0; i < ret.length; i++)
			ret[i] = (actual[i] > 0.5)?1:0;

		return ret;

	}


	/**
	*	"Feeds" input into the <code>Network</code> to
	*	get output(s)
	*	@param inputs 	Array of inputs
	* 	@return Computed ouputs
	*/
	public double[] feed_forward(double[] inputs){

		for(Layer l: this.layers){

			l.set_inputs(inputs);
			inputs = l.feed_forward(l.get_inputs());

		}

		return this.layers.get(this.layers.size() - 1).get_outputs();

	}	


	/**
	*	Computes the error value for a <code>Perceptron</code>
	*	in a Hidden <code>Layer</code>
	*	@param next_l	List of <code>Perceptrons</code> in next <code>Layer</code>
	*	@param index 	Index of the target <code>Perceptron</code>
	*	@param size 	Number of nodes in <code>Layer</code>
	*	@return Error value for <code>Perceptron</code>
	*/
	private double get_error(List<Perceptron> next_l ,int index, int size){

		double errors = 0;
		double[] weights = this.get_weights_to_next(next_l,index,size);

		for(int i = 0; i < size; i++){
			Perceptron p = next_l.get(i);
			errors  += weights[i] * p.error;
		}

		return errors;

	}

	/**
	*	Gets the weights of the edges connected to the node
	*	@param next_l	List of <code>Perceptrons</code> in next <code>Layer</code>
	*	@param index 	Index of the target <code>Perceptron</code>
	*	@param size 	Number of nodes in <code>Layer</code>
	*	@return Weights of the edges connected to target <code>Perceptron</code>	
	*/
	private double[] get_weights_to_next(List<Perceptron> next_l,int index, int size){

		double[] weights = new double[size];
		for(int i = 0; i < size; i++){
			Perceptron p = next_l.get(i);
			weights[i] = p.get_weight_of_input(index);
		}

		return weights;

	}

	/**
	*	For training the <code>Network</code>
	*	@param inputs 	Training set inputs
	*	@param outputs 	Training set traget outputs
	*	@param epochs	Number of training sessions
	*	@param lr 		Learning Rate
	*	@param mtm 		Momentum
	*	@return Trained version of the <code>Network</code>
	*/
	public Network train(double[][] inputs, double[][] outputs, int epochs, double lr, double mtm){

		for(int i = 0; i < epochs;i++)
			for(int j = 0; j < inputs.length; j++){
				
				this.feed_forward(inputs[j]);
				this.back_propagate(outputs[j],lr,mtm);
				
			}

		return this;

	}

}