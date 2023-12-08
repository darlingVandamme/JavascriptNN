import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;

public class NNet {
   private Neuron[] input;
   private Neuron[] output;
   private int layers = 1;
   public double step = 2;
   public int batchSize = 10;
   int batches = 0;
   public int trainings = 0;
   public  int count = 0;
            
    private ArrayList<Neuron> neurons;
    private double costs[] = new double[500];
    public NNet(int inputSize){
        input = new Neuron[inputSize];
        for (int i = 0; i < input.length; i++) {
            input[i] = new Neuron();
        }
        output = input;
        this.layers = 1;
        this.step = 10;
        this.batchSize = 10;
        this.trainings = 0;
        this.count = 0;
        this.neurons = new ArrayList();
    }

    public void addLayer(int size){
        Neuron[] layer = new Neuron[size];
        for (int i = 0; i < size; i++) {
            layer[i] = new Neuron();
            layer[i].connect(this.output);
            this.neurons.add(layer[i]);
        }
        this.output = layer;
        this.layers++;
    }

    public void reset() {
        this.neurons.forEach(n->n.reset());
        /*for (Iterator iterator = this.neurons.iterator(); iterator.hasNext(); ) {
            ((Neuron) iterator.next()).reset();
        }*/
    }    
        
    public void feed(double[] values){
        this.count++;
        this.reset();
        for (int i = 0; i < this.input.length; i++) {
            //Neuron neuron = this.input[i];
            //neuron.setValue(values[i]);
            this.input[i].setValue(values[i]);
            /*this.input[i].value = values[i];
            this.input[i].reset = false;*/
        }
        /*this.neurons.forEach(n->n.getValue());*/
        this.neurons.forEach(n->n.ff ());
        /*Iterator<Neuron> iter = this.neurons.iterator();
        while (iter.hasNext()) {
            (iter.next()).ff();
        }*/
    }

    public double[] getOutput(){
        double[] result = new double[this.output.length];
        for (int i = 0; i < this.output.length; i++) {
            result[i]=this.output[i].getValue();
        }
        return result;
    }

    public double[] check(double[] values){
        this.feed(values);
        return this.getOutput();
    }

        //getCost(result)
    public double getCost(){
        double result = 0;
        for (int i = 0; i < this.output.length; i++) {
            Neuron n = this.output[i];
            result+=(Math.pow(n.expected - n.getValue(),2  ) / (2*this.output.length));
        }
        return result;
    }

    public int getHighest(double[] pattern){
        double high = 0;
        int highIndex =0;
        for (int i = 0; i < pattern.length; i++) {
            if(pattern[i] > high) {
                high=pattern[i];
                highIndex=i;
            }
        }
        return highIndex;
    }

    public void train(double[] values, double[] expected) {
        this.feed(values);
        // store the expected values in the output neurons
        for (int i = 0; i < this.output.length; i++) {
            this.output[i].expected = expected[i];
        }

        double[] output = this.getOutput();
        int index = this.trainings % this.costs.length;
        this.costs[index] = this.getCost();
        this.trainings++;

        /*for (int i = 0; i < this.input.length; i++) {
            Neuron neuron = this.input[i];
            neuron.getDelta();
        }*/
        Iterator iter = this.neurons.iterator();
        while (iter.hasNext()) {
            ((Neuron) iter.next()).getDelta();
        }
        /*ListIterator iter = this.neurons.listIterator(this.neurons.size());
        while (iter.hasPrevious()) {
            ((Neuron) iter.previous()).getDelta();
        }*/
        // adjust weights
        if (this.trainings % this.batchSize == 0) {
            this.batches++;
            Iterator iter2 = this.neurons.iterator();
            while (iter2.hasNext()) {
                ((Neuron) iter2.next()).learn(this.step / this.batchSize);
            }
            //System.out.println(this.getCost());
        }
    }
    public double getAverageCost(){
        double sum =0;
        for (int i = 0; i < costs.length; i++) {
            sum += costs[i];
        }
        return sum / costs.length;
    }
}
