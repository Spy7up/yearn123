import java.util.Stack;

public class CQueue {
    private Stack<Integer> s1;
    private Stack<Integer> s2;

    public CQueue() {
        this.s1 = new Stack<Integer>();
        this.s2 = new Stack<Integer>();
    }

    @Override
    public String toString() {
        return "CQueue{" +
                "s1=" + s1 +
                ", s2=" + s2 +
                '}';
    }

    public void appendTail(int value) {

            this.s1.push(value);

    }

    public int deleteHead() {
        int temp = 0;
        int popNum = 0;

        if (s1.empty()&&s2.empty()){
            return -1;
        }


        while (true){
            temp = s1.pop();
            if (s1.empty()){
                popNum=temp;
                break;
            };
            s2.push(temp);
        }
        while (!s2.empty()){
            temp = s2.pop();
            s1.push(temp);
        }
        return popNum;
    }

    public static void main(String[] args) {
        CQueue c =new CQueue();
        c.appendTail(7);
        c.appendTail(2);
        c.appendTail(3);
        c.appendTail(5);
        c.appendTail(6);
        c.appendTail(4);

        System.out.println(c.deleteHead());

        System.out.println(c.deleteHead());

        System.out.println(c.deleteHead());
        System.out.println(c.deleteHead());
        System.out.println(c.deleteHead());
        System.out.println(c.deleteHead());
    }
}
