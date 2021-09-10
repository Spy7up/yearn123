import com.sun.xml.internal.ws.api.model.wsdl.WSDLOutput;

import java.util.*;

public class Solution_mid {

    //在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix == null){
            return false;
        }
        int b = matrix.length-1;
        int r = matrix[0].length;
        int l = 0;
        while (l<r && b>=0){
            if (matrix[l][b]>target){
                b--;
            }else if (matrix[l][b]<target){
                l++;
            }else {
                return true;
            }
        }
        return false;
    }
    //给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
    public boolean exist(char[][] board, String word) {

        for (int i = 0 ; i<board.length;i++){
            for (int j = 0; j<board[0].length;j++){
                if(dfs(board,word.toCharArray(),i,j,0)) return true;
            }
        }
        return false;
    }
    public boolean dfs(char[][] board,char[] words,int i,int j,int k){
        if (i<0 || i>=board.length || j<0 || j>=board[0].length || board[i][j] != words[k]){
            return false;
        }
        if (k == words.length-1){
            return true;
        }
        board[i][j] = '\0';
        boolean res = dfs(board,words,i-1,j,k+1) || dfs(board,words,i,j-1,k+1)
                || dfs(board,words,i,j+1,k+1) || dfs(board,words,i+1,j,k+1);
        board[i][j] = words[k];
        return res;
    }
    //机器人的运动范围
    public int movingCount(int m, int n, int k) {
        int[][] borad = new int[m][n];
        int count = 0;
        count = dfs1(borad,0,0,k);

        return count;
    }
    public int dfs1(int[][] board,int i,int j,int k){
        if (i<0 || i>=board.length || j<0 || j>=board[0].length || sunbit(i,j)>k || board[i][j] == 1){
            return 0;
        }
        board[i][j] = 1;
        return dfs1(board,i+1,j,k)+dfs1(board,i,j+1,k)+dfs1(board,i-1,j,k)+dfs1(board,i,j-1,k)+1;

    }
    public  int sunbit(int a, int b){
        int c = a/10;
        int d = b/10;
         return (a%10)+(b%10)+c+d;
    }



    //重建二叉树
    int[] preorder;
    Map<Integer,Integer> dic = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
         this.preorder = preorder;
         for (int i = 0;i<inorder.length;i++){
             dic.put(inorder[i],i);
         }
         return recur(0,0,4);
    }
    public TreeNode recur(int root,int left,int right){
        if (left == right){
            return null;
        }
        TreeNode node = new TreeNode(preorder[root]);
        int i = dic.get(preorder[root]);
        node.left = recur(root+1,left,i-1);
        node.right = recur(root+i-left+1,i+1,right);
        return node;
    }
    //删除重复元素
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null){
            return null;
        }
        ListNode cur = head;
        while(cur != null){
            if (cur.val == cur.next.val){
                cur.next = cur.next.next;
            }else {
                cur = cur.next;
            }
        }
        return head;
    }
//切绳子
    public int cuttingRope(int n) {
        if(n<=3){
            return n-1;
        }
        int a = n / 3;
        int b = n % 3;
        if (b == 0){
            return (int) Math.pow(3,a);
        }else if (b == 1 ){
            return (int) Math.pow(3,a-1)*4;
        }
        return (int) Math.pow(3,a)*2;
    }
    //切绳子2
    public int cuttingRope2(int n) {
        if (n < 4 ){
            return n-1;
        }
        long res = 1;
        while (n > 4){
            res = (res * 3)%1000000007;
            n= n - 3 ;
        }
        return (int)(res * n%1000000007);
    }
//快速排序
    public static void sortCore(int[] array, int startIndex, int endIndex) {
    if (startIndex >= endIndex) {
        return;
    }

        int boundary = boundary(array, startIndex, endIndex);

        sortCore(array, startIndex, boundary - 1);
        sortCore(array, boundary + 1, endIndex);
}

    /*
     * 交换并返回分界点
     *
     * @param array
     *      待排序数组
     * @param startIndex
     *      开始位置
     * @param endIndex
     *      结束位置
     * @return
     *      分界点
     */
    private static int boundary(int[] array, int startIndex, int endIndex) {
        int standard = array[startIndex]; // 定义标准
        int leftIndex = startIndex; // 左指针
        int rightIndex = endIndex; // 右指针

        while(leftIndex < rightIndex) {
            while(leftIndex < rightIndex && array[rightIndex] >= standard) {
                rightIndex--;
            }
            array[leftIndex] = array[rightIndex];

            while(leftIndex < rightIndex && array[leftIndex] <= standard) {
                leftIndex++;
            }
            array[rightIndex] = array[leftIndex];
        }

        array[leftIndex] = standard;
        return leftIndex;
    }
//判断是否能分割和相同的子集
    public boolean canPartition(int[] nums) {
        if(nums.length<2){
            return false;
        }
        int sum = 0;
        int max = 0;
        for (int i:nums){
            sum += i;
            max = Math.max(max,i);
        }
        if (sum%2 != 0){
            return false;
        }
        int target = sum/2;
        if (max > target){
            return false;
        }
        boolean[][] dp = new boolean[nums.length][target+1];
        for (int i = 0;i<dp.length;i++){
            dp[i][0] = true;
        }
        for (int i = 1 ;i<dp.length;i++){
            int num = nums[i];
            for (int j = 0;j<dp[0].length;j++)
            if (j >= num){
                dp[i][j] = dp[i-1][j] | dp[i-1][j - num];
            }else {
                dp[i][j] = dp[i-1][j];
            }

        }
        return dp[nums.length-1][target];
    }
    //单词划分
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> dictionary = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length()+1];
        dp[0] = true;
        for (int i = 1;i <= s.length();i++){
            for (int j = 0;j < i;j++){

                    if (dp[j] && dictionary.contains(s.substring(j,i))){
                        dp[i] = true;
                        break;
                    }
            }
        }
        return dp[s.length()];
    }
    //树的子结构
    public boolean isSubStructure(TreeNode A, TreeNode B) {
       if (A == null || B == null){
           return false;
       }
       return recur(A,B) || isSubStructure(A.left,B) || isSubStructure(A.right,B);

    }
    public boolean recur(TreeNode a, TreeNode b){
        if (b == null){
            return true;
        }
        if (a  == null || a.val != b.val){
            return false;
        }
        return recur(a.left,b.left)&&recur(a.right,b.right);
    }
    //快速幂
    public double myPow(double x, int n) {
         if (x == 0){
             return 0;
         }
         long b = n;
         double res = 1.0;
         if (b < 0){
             x = 1/x;
             b = -b;
         }
         while(b > 0){
             if ((b & 1) ==1) res *= x;
             x *= x;
             b >>= 1;
         }
         return res;

    }
    //爬楼梯
    public int climbStairs(int n) {
         int[] dp = new int[n+1];
         dp[0] = 1;
         dp[1] = 1;
        for (int i = 2; i < dp.length; i++) {
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n];
    }
    //最小花费爬楼梯
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
    }
    //复制复杂链表
//    public Node copyRandomList(Node head) {
//        Map<Node,Node> hash = new HashMap();
//        Node cur = head;
//        while(cur != null){
//            hash.put(cur,new Node(cur.val));
//            cur = cur.next;
//        }
//        cur = head;
//        while(cur != null){
//            hash.get(cur).next = hash.get(cur.next);;
//            hash.get(cur).random = hash.get(cur.next);;
//            cur = cur.next;
//        }
//        return hash.get(head);
//    }
//    public Node copyRandomList1(Node head) {
//        if (head == null){
//            return null;
//        }
//        Node cur = head;
//        while(cur != null){
//            Node temp = new Node(cur.val);
//            temp.next = cur.next;
//            cur.next = temp;
//            cur = temp.next;
//        }
//        cur = head;
//        while(cur != null){
//            if (cur.random != null)
//                cur.next.random = cur.random.next;
//            cur.next = cur.next.next;
//        }
//        cur = head.next;
//        Node per = head,res = head.next;
//        while (cur.next != null){
//            per.next = per.next.next;
//            cur.next = cur.next.next;
//            per = per.next;
//            cur = cur.next;
//        }
//        per.next = null;
//        return res;
//    }
    //二叉搜索树和双向链表
    Node pre,head;
    public Node treeToDoublyList(Node root) {
        if (root == null){
            return null;
        }
        dfs(root);
        head.left = pre;
        pre.right = head;
        return head;
    }
    public void dfs(Node root){
      if (root == null){
          return;
      }
      dfs(root.left);
      if (pre != null) pre.right = root;
      else head = root;
      root.left = pre;
      pre = root;
      dfs(root.right);
    }
    //栈的压入和弹出序列
    public boolean validateStackSequences(int[] pushed, int[] popped) {
           Stack<Integer> s = new Stack<>();
           int i = 0;
           for (int n : pushed){
               s.push(n);
               while (!s.empty() || s.peek() == popped[i]){
                  s.pop();
                   i++;
               }
           }
           return s.empty();

    }
    //字符串的排列
    List<String> res = new LinkedList<>();
    char[] c;
    public String[] permutation(String s) {
        c = s.toCharArray();
        dfs(0);
        return res.toArray(new String[res.size()]);
    }
    void dfs(int x) {
        if(x == c.length - 1) {
            res.add(String.valueOf(c));      // 添加排列方案
            return;
        }
        HashSet<Character> set = new HashSet<>();
        for(int i = x; i < c.length; i++) {
            if(set.contains(c[i])) continue; // 重复，因此剪枝
            set.add(c[i]);
            swap(i, x);                      // 交换，将 c[i] 固定在第 x 位
            dfs(x + 1);                      // 开启固定第 x + 1 位字符
            swap(i, x);                      // 恢复交换
        }
    }
    void swap(int a, int b) {
        char tmp = c[a];
        c[a] = c[b];
        c[b] = tmp;
    }
    //从上到下打印二叉树
    public int[] levelOrder(TreeNode root) {
        Queue<TreeNode> que = new LinkedList();
        List<Integer> res = new ArrayList<Integer>();

        if (root == null){
            return new int[0];
        }
        que.add(root);
        while(!que.isEmpty()){
            TreeNode temp = que.poll();
            res.add(temp.val);
            if (temp.left != null){
                que.add(temp.left);
            }
            if (temp.right != null){
                que.add(temp.right);
            }
        }
        int[] res0 = new int[res.size()];
        for (int i = 0; i < res.size(); i++) {
            res0[i] = res.get(i);
        }
        return res0;
    }
    //数字序列中某一位的数字
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        while (n > count) { // 1.
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        long num = start + (n - 1) / digit; // 2.
        return Long.toString(num).charAt((n - 1) % digit); // 3.
    }
    //从上到下打印二叉树 III
    public List<List<Integer>> levelOrder1(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if(root != null) queue.add(root);
        while(!queue.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            for(int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                if(res.size() % 2 == 0) tmp.addLast(node.val); // 偶数层 -> 队列头部
                else tmp.addFirst(node.val); // 奇数层 -> 队列尾部
                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            res.add(tmp);
        }
        return res;
    }
    //二叉树中和为某一值的路径
    List<List<Integer>> res1 = new LinkedList<>();
    LinkedList<Integer> path = new LinkedList<>();
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        if (root == null) {
             return null;
        }
        dfs(root,target);
        return res1;
    }
    public void dfs(TreeNode root,int tar){
        if (root == null){
            return;
        }
        path.add(root.val);
        tar -= root.val;
        if (tar == 0 && root.left == null && root.right == null){
            res1.add(new LinkedList(path));
        }
        dfs(root.left,tar);
        dfs(root.right,tar);
        path.removeLast();
    }
    //数组中数字出现的次数
    public int[] singleNumbers(int[] nums) {
        int m = 1;
        int n = 0;
        for (int num:nums){
            n ^= num;
        }
        while((n & m) == 0){
            m <<= 1;
        }
        int a = 0;
        int b = 0;
        for(int num:nums){
            if ((num & m) == 0){
                a ^= num;
            }else {
                b ^= num;
            }
        }
        return new int[]{a,b};
    }
    //数组中数字出现的次数2
    public int singleNumber(int[] nums) {
        Arrays.sort(nums);
        if (nums[0] != nums[1]){
            return  nums[0];
        }
        if (nums[nums.length-1] != nums[nums.length-2]){
            return  nums[nums.length-1];
        }
        for (int i = 2; i < nums.length; i++) {
            if (nums[i-1] != nums[i] && nums[i+1] != nums[i]){
                return nums[i];
            }

        }
        return 0;
    }
    //把数组排成最小数
    public String minNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for(int i = 0; i < nums.length; i++)
            strs[i] = String.valueOf(nums[i]);
        Arrays.sort(strs, (x, y) -> (x + y).compareTo(y + x));
        StringBuilder res = new StringBuilder();
        for(String s : strs)
            res.append(s);
        return res.toString();
    }
    //把数字翻译成字符串
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < s.length(); i++) {
            String temp = s.substring(i-2, i);
            if(temp.compareTo("10") >= 0 && temp.compareTo("25") <= 0)
                dp[i] = dp[i-1]+dp[i-2];
            else
                dp[i] = dp[i-1];
        }
        return dp[dp.length-1];
    }
    //礼物的最大价值
    public int maxValue(int[][] grid) {
         int[][] dp = new int[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (i == 0 && j == 0){
                    dp[i][j] = grid[i][j];
                }else if (i == 0){
                    dp[i][j] = dp[i][j-1] + grid[i][j];
                }else if (j == 0){
                    dp[i][j] = dp[i-1][j] + grid[i][j];
                }else {
                    dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]) + grid[i][j];
                }
            }
        }
        return dp[grid.length-1][grid[0].length-1];
    }
    //最长不含重复子字符
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() <= 0) {
            return 0;
        }
        int max = 0;
        int left = -1;
        int i = 0;
        HashMap<Character,Integer> map = new HashMap();
        while(i < s.length()){
            if (map.containsKey(s.charAt(i))){
                left = Math.max(left,map.get(s.charAt(i)));
            }
            map.put(s.charAt(i),i);
            max = Math.max(max,i-left);
            i++;
        }
        return max;
    }
    //丑数
    public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        dp[0] = 1;
        int a = 0;
        int b = 0;
        int c = 0;
        for (int i = 0; i < n; i++) {
            dp[i] = Math.min(Math.min(dp[a]*2,dp[b]*3),dp[c]*5);
            if(dp[i] == dp[a]*2) a++;
            if(dp[i] == dp[b]*3) b++;
            if(dp[i] == dp[c]*5) c++;
        }
        return dp[n-1];
    }
    LinkedList<Integer> a = new LinkedList();
    LinkedList<Integer> b = new LinkedList();


    public int max_value() {
        if (b != null){
            return b.poll();
        }
        return -1;
    }

    public void push_back(int value) {
        while (!b.isEmpty() && b.peekLast() < value) {
            b.pollLast();
        }
        b.offerLast(value);
        a.offer(value);


    }

    public int pop_front() {
        if (a.isEmpty()) {
            return -1;
        }
        int ans = a.poll();
        if (!b.isEmpty() && ans == b.peekFirst()) {
            b.pollFirst();
        }
        return ans;
    }
    //构建乘积数组
    public int[] constructArr(int[] a) {
        int len = a.length;
        if(len == 0) return new int[0];
        int[] b = new int[len];
        b[0] = 1;
        int tmp = 1;
        for(int i = 1; i < len; i++) {
            b[i] = b[i - 1] * a[i - 1];
        }
        for(int i = len - 2; i >= 0; i--) {
            tmp *= a[i + 1];
            b[i] *= tmp;
        }
        return b;
    }



    public static void main(String[] args) {
        int a = 1;
        int b = 2;
        int c;
        if ((a++)==3 && b++==3){
            c = 1;
        }

        System.out.println(b);

    }
}
