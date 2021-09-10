import java.util.Arrays;


public class Main {
    public static void main(String[] args) {
        System.out.println(solution("leetcode"));
    }
//    public static String solution(int[] nums){
//        HashMap<Integer,Integer> map = new HashMap();
//        for (int i : nums){
//            if (!map.containsKey(i)){
//                map.put(i,1);
//            }
//        }
//        for (int i = 1; i <= nums.length; i++) {
//            if (map.containsKey(i)){
//                continue;
//            }else {
//                return "No";
//            }
//        }
//        return "Yes";
//    }
//    public static int solution(String s){
//       int [][] dp = new int[s.length()][s.length()];
//        for (int i = 0; i < s.length(); ++i) {
//            Arrays.fill(dp[i],Integer.MAX_VALUE);
//        }
//        return rec(s.toCharArray(),0,s.length()-1,dp);
//    }
//    public static int rec(char [] words,int left ,int right ,int[][] dp){
//        if (left >= right){
//            return 0;
//        }
//        if (dp[left][right] != Integer.MAX_VALUE){
//            return dp[left][right];
//        }
//        if (words[left] == words[right]){
//            return dp[left][right] = rec(words,left+1,right -1,dp);
//        }
//
//        return dp[left][right] = Math.min(rec(words,left+1,right,dp),rec(words,left,right-1,dp))+1;
//    }

        public static int solution(int n,String s,int[] p) {
             int[] dp = new int[n+1];
             dp[0] = 0;
             int res = 0;
            for (int i = 0; i < dp.length; i++) {
                for (int j = i;j<p.length;j++){

                }
            }

        }

}
