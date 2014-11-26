package ai.ann;
import java.util.List;
import java.util.ArrayList;

/**
*	Layer of an Artificial Neural Network
*/

public class Layer implements java.io.Serializable{

	public String name;
	public List<Perceptron> perceptrons;

	private int n_nodes;
	private int n_in;
	private int n_bias;
	private double[] inputs;
	
	public Layer(String name, int n_nodes, int n_in, int n_bias, ActvFunct f_act ){
		this.name = name;
		this.n_nodes = n_nodes;
		this.n_in = n_in;
		this.n_bias = n_bias;
		this.perceptrons = this.init_perceptrons(n_nodes,n_in,n_bias,f_act);

		this.inputs = new double[n_in + n_bias];
	}

	/**
	*	Adjusts the weights of each <code>Perceptron</code>
	* 	@param out_errors	Array of computed errors
	*	@param lr 			Learning Rate
	*	@param mtm			Momentum
	*/
	public void back_propagate(double[] out_errors, double lr, double mtm){
		for(int i = 0; i < out_errors.length; i++){
			Perceptron p = this.perceptrons.get(i);
			double err = out_errors[i];
			
			p.update_weights(err,lr,mtm);
			p.update_output();
			
		}

	}

	/**
	* Layer Feed Forward
	* @param inputs 	Array of input values
	* @return Array of outputs of each <code>Perceptron</code> in <code>Layer</code>
	*/

	public double[] feed_forward(double[] inputs){

		double[] ret = new double[n_nodes];

		int i = 0;
		
		for(Perceptron p: this.perceptrons){
			
			ret[i] = p.get_output(inputs);
			i++;

			
		}

		return ret;		

	}


	/**
	*	Returns inputs
	*	@return Array of inputs
	*/
	public double[] get_inputs(){
		return this.inputs;
	}

	/**
	*	Returns the number of nodes in <code>Layer</code>
	*	@return number of nodes
	*/
	public int get_n_nodes(){
		return this.n_nodes;
	}

	/**
	*	Gets the output of each <code>Perceptron</code> 
	*	in the <code>Layer</code>
	*	@return Output of each <code>Perceptron</code>
	*/
	public double[] get_outputs(){
		double[] outputs = new double[this.n_nodes];
		int i = 0;
		
		for(Perceptron p: this.perceptrons){
			outputs[i] = p.output;
		}

		return outputs;
	}


	/**
	*	Creates an initial list of <code>Perceptron</code>
	*	@param n_nodes 	Number of nodes in the layer
	*	@param n_in		Number of nodes in input Layer
	*	@param n_bias	Number of bias nodes
	*	@param f_act	Activation function
	*	@return List of <code>Perceptron</code>
	*/
	private List<Perceptron> init_perceptrons(int n_nodes, int n_in, int n_bias ,ActvFunct f_act){
		List<Perceptron> ret = new ArrayList<Perceptron>();
		for(int i = 1; i <= n_nodes; i++){
			String name = this.name + "_" + i;
			Perceptron p = new Perceptron(name,f_act, n_in, n_bias);
			ret.add(p);
		}

		return ret;
	}


	/**
	*	Sets the inputs
	*	@param inputs 	Array of new inputs
	*/
	public void set_inputs(double[] inputs){
		this.inputs = inputs;

	}

	/**
	*	Returns string representation of <code>Layer</code>
	*	@return String reperesentation
	*/
	public String toString(){
		String msg = "Layer " + this.name + " with Perceptrons having:\n";
		for(Perceptron p: this.perceptrons)
			msg += p + "\n";
		return msg;
	}
}