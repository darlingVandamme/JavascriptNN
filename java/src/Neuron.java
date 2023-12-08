import java.util.ArrayList;

public class Neuron {
    private static int neurons = 0;
    private int id = (neurons++);
    private Connection[] in ;
    private ArrayList<Connection> out ;
    double value;
    private double delta;
    private double deltaSum;
    private int deltaN;
    private double bias;
    private double z =1;
    private int layer = 0;
    public double expected =0;
    boolean reset=true;

    public Neuron(){ // link to network?
        this.bias = this.getRandom();
        this.z = 1;
        this.layer = 0;
    }

    public void reset(){
        this.value = 0;
        this.delta = 0;
        this.reset = true;
    }

    public void connect(Neuron other[]){
        this.in = new Connection[other.length];
        for(int i=0;i<other.length;i++){
            Connection conn = new Connection(other[i],this,this.getRandom() );
            this.layer = other[i].layer+1;
            other[i].addOut(conn);
            this.in[i]=conn;
        }
    }

    private void addOut(Connection conn){
        if (this.out == null){
            this.out=new ArrayList();
        }
        this.out.add(conn);
    }

    double ff(){
        this.reset=false;
        this.z = this.bias;
        for (Connection conn: this.in){
            this.z += (conn.weight * conn.in.getValue());
        }

        /*for (int i = 0; i < this.in.length; i++) {
            //Connection conn = this.in[i];
            this.z += (this.in[i].weight * this.in[i].in.getValue());
            //System.out.println("add "+this.z+"  "+this.bias+"   "+conn.weight+"   "+conn.in.getValue());
        }*/
        //System.out.println(" z "+this.z+"  "+this.bias);
        this.value = this.activate(this.z); // sigmoid
        return this.value;
    }

    public double getValue(){
        if (this.reset){
            this.ff();
        }
        return this.value;
    }

    public void setValue(double value){
        this.value=value;
        this.reset=false;
    }

    public double getDelta(){
        if (this.isInput()) return 0;
        // lazy calculate
        if (this.delta==0) {
            double zDeriv = this.actDeriv(this.z);
            if (this.isOutput() ){
                this.delta = ((  this.value - this.expected )  * zDeriv );
                //System.out.println("out "+this.delta);
            } else {
                // other neurons
                double sum = 0;
                /*
                Iterator<Connection> it = this.out.iterator();
                while (it.hasNext()) {
                    Connection conn = it.next();
                    sum += (conn.out.getDelta() * conn.weight );
                }*/
                for(Connection conn : out){
                    sum += (conn.out.getDelta() * conn.weight );
                }

                /*double sum = this.out.stream().reduce(
                       0.0, (accum, conn)-> accum + (conn.out.getDelta() * conn.weight), Double::sum);
                 */


                this.delta = sum * zDeriv;
                //System.out.println("mid "+sum+" "+zDeriv+" "+this.z+"   "+this.delta);
            }
            for (int i=0;i<this.in.length;i++) {
                this.in[i].addDelta(this.delta);
                //System.out.println("mid "+this.in[i].delta);
            }
            this.deltaSum += this.delta;
            this.deltaN++;
            //System.out.println("mid "+this.deltaSum);
        }
        return this.delta;
    }

    public void learn(double step) {
        if (this.deltaN > 0) {
            if (!this.isInput()) {
                this.bias -= step * this.deltaSum;
                for (int i=0;i<this.in.length;i++) {
                    this.in[i].learn(step);
                }
            }
            // reset delta
            this.deltaSum = 0;
            this.delta = 0;
            this.deltaN=0;
        }
    }

    public boolean isInput(){
        return this.in == null;
    }

    public boolean isOutput(){
        return this.out == null;
    }

    public double getRandom(){
        // return (Math.random()*4)-2
        // normal distribution
        double u1 = Math.random();
        double u2 = Math.random();
        double dev = 1;
        return  ( Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) / dev );
    }

    public double activate(double x){
        return sigmoid(x);
    }
    public double sigmoid(double x){
        return 1.0 / (1.0 + Math.exp(-x));
    }
    public double actDeriv(double x){
        return sigmoidDeriv(x);
    }
    public double sigmoidDeriv(double x){
        //if (!x) return 1
        double s = sigmoid(x);
        return s * (1-s);
    }

}

class Connection {
    Neuron in;
    Neuron out;
    double weight=0;
    double deltaSum=0;
    Connection(Neuron in, Neuron out, double weight){
        this.in = in;
        this.out=out;
        this.weight = weight;
    }
    void learn(double step){
        this.weight -= step * this.deltaSum;
        this.deltaSum=0;
    }
    void addDelta(double delta){
        this.deltaSum += delta * this.in.getValue();
    }
}
