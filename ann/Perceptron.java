package ai.ann;
import java.util.Random;

/**
*	Elementary unit of an Artificial Neural Network
*/
public class Perceptron implements java.io.Serializable{

	public String name;
	public double output;
	public double error; //Small delta
	
	private ActvFunct f_act; //Activation function
	private ActvFunct f_err; //Derivative of the activation function
	private int n_in; //Number of inputs
	private int n_bias; //Number of bias
	private double[] weights; //Array of Weights from bias and inputs, first n elements are for biases
	private double[] inputs; //Outputs of input nodes
	private double[] d_weights; //Change of weight

	public Perceptron(String name, ActvFunct f_act, int n_in, int n_bias){
		double h = 1e-6;

		this.name = name;
		this.f_act = f_act;
		//Get derivative of activation function
		this.f_err = (double net) -> ((f_act.activate(net + h/2) - f_act.activate(net - h/2))/h);
		this.n_in = n_in;
		this.n_bias = n_bias;
		this.weights = this.init_weight(n_in,n_bias);
		
		this.d_weights = new double[n_in + n_bias];
		this.inputs = new double[n_in];
	}

	/**
	* Returns the output of the <code>Perceptron</code> given the inputs 
	* @param	inputs 	Array of outputs of input nodes
	* @return 	Output of the <code>Perceptron</code>
	*/
	public double get_output(double inputs[]){
		
		this.inputs = inputs;
		this.update_output();
		return output;
	}

	/**
	*	Gets the net weights multiplied to
	*	corresponding node output
	*	@return Net weight
	*/
	private double get_total_weight(){
		double total = 0;

		for(int i = 0; i < this.n_bias; i++)
			total += this.weights[i] * 1;

		for(int i = 0; i < this.n_in; i++)
			total += this.weights[i + n_bias] * this.inputs[i];
		
		return total;
	}

	/**
	*	Returns the weight for some input node
	*	@param index					Index of input node
	*	@return Weight for some input node
	*/
	public double get_weight_of_input(int index){
		return this.weights[index];
	}

	/**
	*	Returns weights from bias and inputs
	*	@return Array of Weights from bias and inputs 
	*/
	public double[] get_weights(){
		return this.weights;
	}

	/**
	*	Puts a random number between -0.5 and 0.5 as
	*	initial values for the weights
	*	@param n_in		Number of input nodes
	*	@param n_bias	Number of bias nodes 
	*	@return Array of random numbers of size
	*					n_in + n_bias
	*/
	private double[] init_weight(int n_in,int n_bias){
		double[] array = new double[n_in + n_bias];
		Random rand_gen = new Random();
		for(int i = 0; i < n_in + n_bias; i++)
			array[i] = rand_gen.nextDouble() - 0.5;

		return array;

	}

	/**
	*	Puts a 0.5 and 0.4 alternatingly as initial
	*	value for weights. Used for testing.
	*	@param n_in		Number of input nodes
	*	@param n_bias	Number of bias nodes 
	*	@return Array of alternating 0.5 and 0.4 values
	*/
	private double[] init_weight_test(int n_in, int n_bias){

		double[] array = new double[n_in + n_bias];

		for(int i = 0; i < n_in + n_bias; i++)
			if(i%3 == 0 || i%3 == 2)
				array[i] = 0.5;
			else
				array[i]= 0.4;

		return array;

	}

	/**
	*	Returns string representation of <code>Perceptron</code>
	*	@return <code>String</code> reperesentation
	*/
	public String toString(){
		String msg = this.name + " with " + this.n_in + " input nodes and " + this.n_bias + " bias node(s)";
		return msg;
	}

	/**
	*	Updates the output of the <code>Perceptron</code>
	*/
	public void update_output(){

		double t_weights = this.get_total_weight(); 
		this.output = this.f_act.activate(t_weights);
	
	}	

	/**
	*	Updates the weights of the <code>Perceptron</code> based on computed error 
	*	@param error	Computed error
	*	@param lr		Learning rate
	*	@param mtm 		Momentum
	*/
	public void update_weights(double error, double lr, double mtm){
		
		double p_err = this.f_err.activate(this.get_total_weight());
		this.error = error * p_err;

		for(int j = 0; j < this.n_bias; j++){
			double last_d = this.d_weights[j];
			this.d_weights[j] = ((1-mtm) * lr * this.error) + (mtm*last_d);
			this.weights[j] += this.d_weights[j]; 
		}

		for(int j = 0; j < this.inputs.length; j++){
			double last_d = this.d_weights[j + this.n_bias];
			this.d_weights[j + this.n_bias] = ((1-mtm) * lr * this.error * this.inputs[j]) + (mtm*last_d);
			double holder = this.weights[j + this.n_bias];
			this.weights[j + this.n_bias] += this.d_weights[j + this.n_bias];
		}

	}


}