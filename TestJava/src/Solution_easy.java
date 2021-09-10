import java.util.*;

public class Solution_easy {
    //斐波那契数列
    public static long fib(int n) {
        int[] nums = new int[n+1];

        if(n==0){
            return 0;
        }else{
            nums[0] = 1;
            nums[1] = 1;
            for(int i = 2;i<=n;i++){
                nums[i]=(nums[i-1]+nums[i-2])%1000000007;
            }
        }
        return nums[n];
    }
    //旋转数组的最小数字（二分法）
    public int minArray(int[] numbers) {
        int i=0;
        int j=numbers.length-1;
        while (i<j){
            int m=i+j/2;
            if (numbers[m]<numbers[j]){
                j=m;
            }else if(numbers[m]>numbers[j]){
                i=m+1;
            }else{
                j--;
            }
        }
        return numbers[j];
    }
    //替换数组中的空格
    public String replaceSpace(String s) {
          int length = s.length();
          char[] chars = new char[length*3];
          int size=0;
          for(int i=0;i<length;i++){
              char c = s.charAt(i);
              if(c == ' '){
                  chars[size++]='%';
                  chars[size++]='2';
                  chars[size++]='0';
              }else{
                  chars[size++]=c;

              }
          }
          String s1 = new String(chars,0,size);
          return s1;
    }
//链表倒序输出（栈）
    public int[] reversePrint(ListNode head) {
        Stack<ListNode> stack = new Stack<ListNode>();
        ListNode temp = head;
        while (temp != null) {
            stack.push(temp);
            temp = temp.next;
        }
        int size = stack.size();
        int[] print = new int[size];
        for (int i = 0; i < size; i++) {
            print[i] = stack.pop().val;
        }
        return print;
    }
    //合并两个有序链表(保持递增顺序)（伪头指针）
    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
           ListNode headNode = new ListNode(0);
           ListNode cur = headNode;
           while (l1!=null&&l2!=null){
                if (l1.val<l2.val){
                    cur.next=l1;
                    l1=l1.next;

                }else {
                    cur.next=l2;
                    l2=l2.next;

                }
               cur=cur.next;
           }
        int k;
       cur.next = l1 != null ? l1 : l2;

        headNode=headNode.next;
           return headNode;

    }
//镜像二叉树
        public TreeNode mirrorTree(TreeNode root) {
                if (root == null){
                    return root;
                }
                TreeNode temp;
                temp = root.left;
                root.left = root.right;
                root.right = temp;
               if(root.left!=null){
                   mirrorTree(root.left);
               }
               if (root.right!=null){
                   mirrorTree(root.right);
               }
               return root;
        }
        //判断对称二叉树（递归）
    public boolean isSymmetric(TreeNode root) {
        return root == null ? true : recur(root.left, root.right);
    }
    public boolean recur(TreeNode L,TreeNode R){
        if (L==null && R==null){
            return true;
        }
        if (L==null || R ==null || R.val!=L.val){
            return false;
        }
        return recur(L.left, R.right) && recur(L.right, R.left);
    }

//奇数位于数组前面（栈）
//    public int[] exchange(int[] nums) {
//          Stack<Integer> s = new Stack<Integer>();
//          for(int i:nums){
//              if(i%2==0){
//                  s.push(i);
//              }
//          }
//        for(int i:nums){
//            if(i%2!=0){
//                s.push(i);
//            }
//        }
//        for (int i=0 ;i<nums.length ;i++){
//            nums[i]=s.pop();
//        }
//        return nums;
//    }
//奇数位于数组前面（双指针）
public int[] exchange(int[] nums) {
        int i = 0;
        int j = nums.length-1;
   while (i<j){
       if (nums[i]%2==1){
           i++;
       }else{
           while(j>i){
               if (nums[j]%2==1){
                   int temp;
                   temp = nums[i];
                   nums[i] = nums[j];
                   nums[j] = temp;
                   break;
               }
               j--;
           }
       }
   }
    return nums;
}
//二进制中的1的个数
    public int hammingWeight(int n) {
        int ret = 0;
        while (n != 0) {
            n &= n - 1;
            ret++;
        }
        return ret;
    }

    public int[] spiralOrder1(int[][] matrix) {
        if(matrix.length == 0) return new int[0];
        int l = 0, r = matrix[0].length - 1, t = 0, b = matrix.length - 1, x = 0;
        int[] res = new int[(r + 1) * (b + 1)];
        while(true) {
            for(int i = l; i <= r; i++) res[x++] = matrix[t][i]; // left to right.
            if(++t > b) break;
            for(int i = t; i <= b; i++) res[x++] = matrix[i][r]; // top to bottom.
            if(l > --r) break;
            for(int i = r; i >= l; i--) res[x++] = matrix[b][i]; // right to left.
            if(t > --b) break;
            for(int i = b; i >= t; i--) res[x++] = matrix[i][l]; // bottom to top.
            if(++l > r) break;
        }
        return res;
    }

//顺时针输出数组
    public static int[] spiralOrder2(int[][] matrix) {
        if(matrix.length==0){
            return new int[0];
        }
        int l = 0;
        int t = 0;
        int r = matrix[0].length-1;
        int b = matrix.length-1;
        int x = 0;
        int[] res = new int[(r+1)*(b+1)];
        while (true){
            for (int i = l;i<=r;i++){res[x++] = matrix[t][i];}
            if (++t > b) break;
            for (int i = t;i<=b;i++){res[x++] = matrix[i][r];}
            if (--r < l) break;
            for(int i = r ;i>=l;i--){res[x++] = matrix[b][i];}
            if (--b < t) break;
            for(int i = b ;i>=t;i--){res[x++] = matrix[i][l];}
            if (++l > r) break;
        }
        return res;
    }
//输出链表倒数k个节点
    public ListNode getKthFromEnd(ListNode head, int k) {
          ListNode p1 = head;
          ListNode p2 = p1;
          int num=0;
          while(p1!=null){
              num=0;
              p2=p1;
              while(p2!=null){
                  num++;
                  p2=p2.next;
              }
              if(num == k){break;}
              p1=p1.next;
          }
          return p1;
    }
    //从1输出k位最大数
    public int[] printNumbers(int n) {
        int sum = 1;
        while(n>0){
            n--;
            sum = sum*10;
        }
        int[] nums = new int[sum];
        for(int i = 1;i<sum;i++){
           nums[i]=i;
        }
        return nums;

    }
//反转链表
    public ListNode reverseList(ListNode head) {
     ListNode prev = null;
     ListNode curr = head;
     while (curr!=null){
         ListNode next = curr.next;
         curr.next = prev;
         prev = curr;
         curr = next;
     }
     return prev;
    }
    //删除链表中指定值
    public ListNode deleteNode(ListNode head, int val) {
        ListNode p1 = head;
        ListNode p2 = p1;
        while(p1 != null){
            if (p1.val != val){
                p2=p1;
                p1 = p1.next;
            }else {
                if (p1 == head){
                    p1 = head.next;
                    p2=p1;
                    head = p1;
                }
               p2.next = p1.next;
                p1 = p1.next;

            }
        }
        return head;
    }
    //最小的k个数（堆）

        public int[] getLeastNumbers(int[] arr, int k) {
            if (k == 0 || arr.length == 0) {
                return new int[0];
            }
            // 默认是小根堆，实现大根堆需要重写一下比较器。
            Queue<Integer> pq = new PriorityQueue<>((v1, v2) -> v2 - v1);
            for (int num: arr) {
                if (pq.size() < k) {
                    pq.offer(num);
                } else if (num < pq.peek()) {
                    pq.poll();
                    pq.offer(num);
                }
            }

            // 返回堆中的元素
            int[] res = new int[pq.size()];
            int idx = 0;
            for(int num: pq) {
                res[idx++] = num;
            }
            return res;
        }

//子数组的最大和
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        for (int i =1 ; i<nums.length; i++){
            nums[i] = nums[i]+Math.max(nums[i-1],0);
            res = Math.max(res,nums[i]);
        }
        return res;
    }
    //出现次数为数组的一半的数字
    public int majorityElement(int[] nums) {
        int x = 0;
        int vote = 0;
        for(int i :nums){
            if (vote == 0 ){
                x = i;
            }
            vote += x == i ? 1:-1;
        }
        return x;
    }

    //合并区间
    public int[][] merge(int[][] intervals) {
        // 1. 按照区间左边的值进行升序排列
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        // 2. 初始化 outputs, 用于存储合并之后区间结果的动态数组
        ArrayList<int[]> outputs = new ArrayList<>();
        // 3. 遍历处理每一个区间
        for (int i = 0; i < intervals.length; i++) {
            int[] currInterval = intervals[i];
            if (outputs.isEmpty()) {
                outputs.add(currInterval);
            } else { // 判断是否有重叠，有的话则合并
                int[] outputsLastInterval = outputs.get(outputs.size() - 1);
                int outputLastRight = outputsLastInterval[1];
                int currLeft = currInterval[0];
                if (outputLastRight < currLeft) { // 没有重叠
                    outputs.add(currInterval);
                } else { // 重叠，合并
                    int currRight = currInterval[1];
                    outputsLastInterval[1] = Math.max(outputLastRight, currRight);
                }
            }
        }
        return outputs.toArray(new int[outputs.size()][]);
    }
    //从上到下打印树
    public List<List<Integer>> levelOrder(TreeNode root) {
          List<List<Integer>> res = new ArrayList<List<Integer>>();
          Queue<TreeNode> que = new LinkedList<TreeNode>();
          if (root!=null){que.add(root);}
          while (!que.isEmpty()){
              List<Integer> temp = new ArrayList<>();
              for (int i = que.size();i>0;i--){
                  TreeNode a = que.poll();
                  temp.add(a.val);
                  if (a.left!=null){que.add(a.left);}
                  if (a.right!=null){que.add(a.right);}
              }
              res.add(temp);
          }
              return res;

          }
//第一个只出现一次的的字符
    public char firstUniqChar(String s) {
        Map<Character,Integer> map =new  HashMap<Character,Integer>();
        char c;
        for (int i = 0;i<s.length();i++){
            c = s.charAt(i);
            map.put(c,map.getOrDefault(c,0)+1);
        }
        for (int i = 0;i<s.length();i++){
           if (map.get(s.charAt(i)) == 1){
               return s.charAt(i);
           }
        }
         return ' ';
    }
    //寻找两个链表的第一个公共节点
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA;
        ListNode B = headB;
        while(A != B ){
            A = A!=null ? A.next :headB;
            B = B!=null ? B.next :headA;
        }
        return A;

    }

    //二叉树的深度
    //递归
    public int maxDepth(TreeNode root) {
       if (root == null){
           return 0;
       }
       return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }
    //层序遍历
    public int maxDepth1(TreeNode root) {
        int res=0;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        if (root != null){
            queue.add(root);
        }else {
            return 0;
        }
            while (!queue.isEmpty()){
                Queue<TreeNode> temp = new LinkedList<TreeNode>();
                for(TreeNode node : queue) {
                    if(node.left != null) temp.add(node.left);
                    if(node.right != null) temp.add(node.right);
                }
                res++;
                queue = temp;
            }
        return res;
    }
//输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
    public int[] twoSum(int[] nums, int target) {
        int r=nums.length-1;
        int l=0;
        int[] res = null;
        if (nums[0]>target){return res;}
        while(l<r){
            int sum = nums[l]+nums[r];
            if (sum > target){
                r--;
            }else if(sum < target){
                l++;
            }else {
                res = new int[2];
                res[0]=nums[l];
                res[1]=nums[r];
                break;
            }
        }
        return res;
    }

    //和为x的连续正数序列
    public int[][] findContinuousSequence(int target) {
        int i = 1;
        List<int[]> res = new ArrayList<int[]>();
        while (i<target){
            int[] temp;
            int sum = 0;
            int k = i;
            while (true){
                sum += k;
                if (sum ==target ){
                    temp = new int[k-i+1];
                    for (int j = 0;j<temp.length;j++){
                        temp[j]=i+j;
                    }
                    res.add(temp);
                    break;
                }else if (sum > target){
                    break;
                }
                k++;
            }
            i++;
        }

        return  res.toArray(new int[res.size()][]);
    }

    //翻转单词顺序
    public String reverseWords(String s) {
        s = s.trim(); // 删除首尾空格
        StringBuffer res = new StringBuffer();
        int j = s.length()-1,i=j;
        while (i >= 0){
            while(i >= 0&&s.charAt(i) != ' ') i--;
            res.append(s.substring(i+1,j+1)+' ');
            while(i >= 0&&s.charAt(i) == ' ') i--;
            j = i;

        }
        return res.toString().trim();
    }
//统计一个数字在排序数组中出现的次数。
    public int search(int[] nums, int target) {
       if (nums[0] > target || nums[nums.length-1] < target){
           return 0;
       }
       int count = 0;
       for (int i = 0;i<nums.length;i++){
           if (nums[i]==target){
               count++;
           }else {
               break;
           }
       }
       return count;
    }
    //左旋转数组
    public String reverseLeftWords(String s, int n) {
        Queue<Character> queue = new LinkedList<Character>();
        char[] res = new char[s.length()];
        int k = 0;
        for (int i = n;i<s.length();i++){
            queue.add(s.charAt(i));
        }
        for (int i = 0;i<n;i++){
            queue.add(s.charAt(i));
        }
        while (!queue.isEmpty()){
            res[k++] = queue.poll();
        }
        return new String(res);
    }
 //0-n中缺失的数字
    public int missingNumber(int[] nums) {
        int i = 0;
         for (;i<nums.length;i++){
             if (i!=nums[i]){
                 break;
             }
         }
         return i;
    }
//不用+，-，*，/，实现加法
    public int add(int a, int b) {
        if(b == 0){
            return a;
        }
        return add(a^b,(a&b)<<1);
    }
    //扑克牌中的顺子
    public boolean isStraight(int[] nums) {
        int joke = 0;
        Arrays.sort(nums);
        for (int i = 0; i<nums.length;i++){
            if (nums[i] == 0){
                joke++;
            }else if (nums[i] == nums[i+1]){
                return false;
            }
        }
        return nums[4] - nums[joke] < 5;
    }
    //判断平衡二叉树
    public boolean isBalanced(TreeNode root) {
          if (root == null){
              return true;
          }
          return Math.abs(height(root.left)-height(root.right))<=1 && isBalanced(root.left) && isBalanced(root.right);
    }

    public int height(TreeNode root){
        if (root == null){
            return 0;
        }
        return Math.max(height(root.left),height(root.right))+1;
    }
    //自下向上递归
//    class Solution {
//        public boolean isBalanced(TreeNode root) {
//            return height(root) >= 0;
//        }
//
//        public int height(TreeNode root) {
//            if (root == null) {
//                return 0;
//            }
//            int leftHeight = height(root.left);
//            int rightHeight = height(root.right);
//            if (leftHeight == -1 || rightHeight == -1 || Math.abs(leftHeight - rightHeight) > 1) {
//                return -1;
//            } else {
//                return Math.max(leftHeight, rightHeight) + 1;
//            }
//        }
//    }



//0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。
    public int lastRemaining(int n, int m) {
          int res = 0;
          for (int i = 2;i<=n;i++){
              res = (res+m)%i;
          }
          return res;
    }
    //给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val == p.val){
            return p;
        }
        if (root.val == q.val){
            return q;
        }
        if ((p.val<root.val && q.val>root.val) || (p.val>root.val && q.val<root.val)){
            return root;
        }else if ((p.val<root.val && q.val<root.val)){
            return lowestCommonAncestor(root.left,p,q);
        }else if ((p.val>root.val && q.val>root.val)){
            return lowestCommonAncestor(root.right,p,q);
        }
        return null;
    }

//    给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null){
            return null;
        }
        if (root == p || root == q){
            return root;
        }
        TreeNode right = lowestCommonAncestor1(root.left,p,q);
        TreeNode left = lowestCommonAncestor1(root.left,p,q);
        if (right != null && left != null){
            return root;
        }
        if (right != null){
            return left;
        }
        if (left != null){
            return right;
        }
        return null;
    }





}


