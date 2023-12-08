import java.io.IOException;
import java.util.Arrays;

public class MNIST {
    private NNet net;
    private MnistMatrix[] train;
    private MnistMatrix[] test;
    private int epochs = 3;
    long trainTime =0;
    public MNIST(){
        this.readData();
        this.net =net = new NNet(28*28);
        net.step = 3;
        net.batchSize = 10;
        net.addLayer(30);
        net.addLayer(10);  // output
    }

    public double[] expected(int value){
        double[] result = new double[10];
        result[value]=1;
        return result;
    }

/*    public double[] translate(int[] values){
        double[] result = new double[values.length];
        for (int j = 0; j < result.length; j++) {
            result[j] = ((double)train[j].getPixels()[j]);
        }
        return result;
    }*/
    public void train(){
        long startTime = System.currentTimeMillis();
        for (int epochs=0;epochs<this.epochs ; epochs++) {
            //net.step = net.step*0.9
            for (int i = 0; i < train.length; i++) {
                double[] input = train[i].getPixels();
                double[] expected = expected(train[i].getLabel());
                net.train(input,expected);
            /*if (net.trainings%5000==0){
                System.out.println("train "+net.trainings+"  Batches "+net.batches+"  Cost "+net.getAverageCost());
            }*/

            }
            check(1000,false);
            System.out.println("epoch "+epochs+"  Cost "+net.getAverageCost() );
            trainTime = System.currentTimeMillis() - startTime;
            System.out.println("trainTime "+trainTime);
        }
    }

    public void check(int size, boolean show){
        long startTime = System.currentTimeMillis();
        int start = (int)(Math.random()*(10000 - size));
        if (show) System.out.println("test images "+start+" "+size );
        MnistMatrix[] testImages = Arrays.copyOfRange(test, start,start+size);
        int correct = 0;
        for (int i=0;i<testImages.length;i++) {
            double[] output = net.check(testImages[i].getPixels());
            int result = net.getHighest(output);
            if(testImages[i].getLabel() == result) correct++;

            /*if (show) System.out.println(((testImages[i].getLabel() == result)?"check ":"MISS ")
                    +i+" "+testImages[i].getLabel()+"  <=> "+result+"  "+ Arrays.toString(output));
             */
        }

        if (show) System.out.println("Training iterations "+ net.trainings+"  TrainTime "+trainTime+" "+((double)net.trainings/(trainTime/1000))+" Trainings/s " + net.step +" step");
        if (show) System.out.println("Training step:"+  net.step +"   BatchSize: "+net.batchSize);
        if (show) System.out.println("Check iterations "+ size+" "+(1000*size/(System.currentTimeMillis()-startTime))+" Checks/s ");
        System.out.println("success rate "+ ((1.0*correct)/testImages.length));
        if (show) System.out.println("Avg Cost "+ net.getAverageCost());
        //if (show) console.log("labels ",net.patterns)
        if (show) System.out.println("check Time "+(System.currentTimeMillis()-startTime));
    }
    public void readData(){
        try {
            train = new MnistDataReader().readData("../../MNIST/train-images.idx3-ubyte", "../../MNIST/train-labels.idx1-ubyte");
            test = new MnistDataReader().readData("../../MNIST/t10k-images.idx3-ubyte", "../../MNIST/t10k-labels.idx1-ubyte");
            System.out.println("read data "+train.length+" "+test.length);
        } catch (IOException e){
            System.out.println("Error reading data "+e);
        }
    }

    public void test(){
        int item = 0;
        double[] input = train[item].getPixels();
        //double[] translated = translate(input);
        System.out.println("item label "+train[item].getLabel());
        for(int i=0;i<28;i++) {
            System.out.println(Arrays.toString(Arrays.copyOfRange(input,i*28,((i+1)*28))));
        }
        /*for(int i=0;i<28;i++) {
            System.out.println(Arrays.toString(Arrays.copyOfRange(translated,i*28,((i+1)*28))));
        }*/
        System.out.println(" expected "+Arrays.toString(expected(train[item].getLabel())));
    }


    public static void main(String[] args) {
        MNIST m = new MNIST();

        // m.test();
        m.train();
        m.check(10000,true);



    }


}
