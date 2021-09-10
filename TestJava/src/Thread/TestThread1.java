package Thread;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.locks.ReentrantLock;

public class TestThread1 implements Runnable {

    private int tickets = 10;
    private final ReentrantLock lock = new ReentrantLock();

    @Override
    public void run(){
        while (true){
            try {
                lock.lock();
            if (tickets > 0){
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println(tickets--);
            }else {
                break;
            }}finally {
                lock.unlock();
            }
        }
    }


}
class Test {
    public static void main(String[] args) {
        TestThread1 t = new TestThread1();
        new Thread(t).start();
        new Thread(t).start();
        ExecutorService a = Executors.newCachedThreadPool();
        new Thread(t).start();

    }
}