import java.util.Arrays;

public class Classifier1 {

    //console.time("total")
    private NNet net;

    public Classifier1() {
        net = new NNet(5);
        net.addLayer(50);
        net.addLayer(2);
        net.batchSize = 10;
        net.step = 2;
    }

    public void train(int iter) {
        /*net.feed(new double[]{1.0,0.0,0.0,2.0,1.0});
        System.out.println(net.getOutput());*/
        long startTime = System.currentTimeMillis();
        for (int i = 1; i < iter; i++) {
            net.train(new double[]{1, 0, 0, 2, 1}, new double[]{0, 1});
            net.train(new double[]{1, 0, 0, 2, 0}, new double[]{0, 1});
            net.train(new double[]{0, 1, 0, 0, 1}, new double[]{1, 0});
            net.train(new double[]{0, 1, 0, 0, 0}, new double[]{1, 0});
            net.train(new double[]{1, 0, 0, 2, 1}, new double[]{0, 1});
        }
        System.out.println("train " + (System.currentTimeMillis() - startTime));
    }

    public void test(double[] values, String expected) {
        net.feed(values);
        System.out.println("expected   "+expected+"  "+ Arrays.toString(this.net.getOutput()));
    }

    public static void main(String[] args) {
        Classifier1 c = new Classifier1();
        c.train(100000);
        c.test(new double[]{1,0,0,2,1},"[0,1]");
        c.test(new double[]{1,0,0,2,0},"[0,1]");
        c.test(new double[]{0,1,0,0,0},"[1,0]");
        c.test(new double[]{0,0,0,0,0},"[???]");
        long startTime = System.currentTimeMillis();
        for (int i=0;i<1000000;i++){
            c.net.feed(new double[]{0,1,0,0,0});
            c.net.getOutput();
        }
        System.out.println("Checks "+(System.currentTimeMillis()-startTime));
        System.out.println(c.net.trainings+"   "+c.net.count);
    }
}
