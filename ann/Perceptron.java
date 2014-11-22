package ai.ann;
import java.util.Random;

public class Perceptron implements java.io.Serializable{
 
	public String name;
	public double output;
	public double error;
	
	private ActvFunct f_act;
	private ActvFunct f_err;
	private int n_in;
	private int n_bias;
	private double[] weights;
	private double[] inputs;
	private double[] d_weights;

	public Perceptron(String name, ActvFunct f_act, int n_in, int n_bias){
		double h = 1e-6;

		this.name = name;
		this.f_act = f_act;
		this.f_err = (double net) -> ((f_act.activate(net + h/2) - f_act.activate(net - h/2))/h);
		this.n_in = n_in;
		this.n_bias = n_bias;
		this.weights = this.init_weight(n_in,n_bias);
		
		this.d_weights = new double[n_in + n_bias];
		this.inputs = new double[n_in];
	}

	public double get_output(double inputs[]){
		
		this.inputs = inputs;
		this.update_output();
		return output;
	}

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

	public double[] get_weights(){
		return this.weights;
	}

	public void update_output(){

		double t_weights = this.get_total_weight(); 
		this.output = this.f_act.activate(t_weights);
	
	}

	private double[] init_weight(int n_in,int n_bias){
		double[] array = new double[n_in + n_bias];
		Random rand_gen = new Random();
		for(int i = 0; i < n_in + n_bias; i++)
			array[i] = rand_gen.nextDouble() - 0.5;

		return array;

	}

	private double[] init_weight_test(int n_in, int n_bias){

		double[] array = new double[n_in + n_bias];

		for(int i = 0; i < n_in + n_bias; i++)
			if(i%3 == 0 || i%3 == 2)
				array[i] = 0.5;
			else
				array[i]= 0.4;

		return array;

	}

	private double get_total_weight(){
		double total = 0;

		for(int i = 0; i < this.n_bias; i++)
			total += this.weights[i] * 1;

		for(int i = 0; i < this.n_in; i++)
			total += this.weights[i + n_bias] * this.inputs[i];
		
		return total;
	}

	public double get_total_weight_test(){
		double total = 0;
		
		for(int i = 0; i < this.n_bias; i++){
			System.out.println("Weight on Bias = " + weights[i] + " on Perceptron " + name);
			total += this.weights[i] * 1;
		}

		for(int i = 0; i < this.n_in; i++){
			System.out.println("Weight = " + weights[i + n_bias] + " Input =" + this.inputs[i] + " on Perceptron " + name);
			total += this.weights[i + n_bias] * this.inputs[i];
		}
		
		return total;
	}

	public double get_weight_of_input(int index){
		return this.weights[index];
	}

	public String toString(){
		String msg = this.name + " with " + this.n_in + " input nodes and " + this.n_bias + " bias node(s)";
		return msg;
	}

}