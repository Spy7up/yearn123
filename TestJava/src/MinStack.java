import java.util.Stack;

class MinStack {
    Stack<Integer> A;
    Stack<Integer> B;
    /** initialize your data structure here. */
    public MinStack() {
      this.A = new Stack<Integer>();
      this.B = new Stack<Integer>();
    }

    public void push(int x) {
       A.add(x);
       if (B.empty() || B.peek()>=x){
           B.add(x);
       }
    }

    public void pop() {
        if(A.pop().equals(B.peek())){
            B.pop();
        }
    }

    public int top() {
        return A.peek();
    }

    public int min() {
        return B.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */