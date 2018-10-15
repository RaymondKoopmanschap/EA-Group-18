import java.io.PrintStream;
import org.vu.contest.ContestSubmission;

public class ConsertTestBox
{
  public ConsertTestBox() {}
  
  public static void main(String[] paramArrayOfString)
  {
    String str1 = null;String str2 = null;
    long l1 = -1L;long l2 = -1L;
    int i = 0;
    for (String str3 : paramArrayOfString) {
      if (str3.startsWith("-submission=")) { str1 = str3.split("=")[1];
      } else if (str3.startsWith("-evaluation=")) { str2 = str3.split("=")[1];
      } else if (str3.startsWith("-seed=")) { l2 = Long.parseLong(str3.split("=")[1]);
      } else if (str3.equals("-nosec")) i = 1; else {
        System.out.println("Invalid flag: '" + str3 + "' !");
      }
    }
    if (str1 == null) throw new Error("Submission ID was not specified! Cannot run...\n Use -submission=<classnamehere> to specify the name of the algorithm class.");
    if (str2 == null) throw new Error("Evaluation ID was not specified! Cannot run...\n Use -evaluation=<classnamehere> to specify the name of the evaluation class.");
    if (l2 < 0L) { throw new Error("Seed was not specified! Cannot run...\n Use -seed=<number> to specify the random seed.");
    }
    
    boolean a = null;
    try {
      a = Class.forName(str2);
    }
    catch (Throwable localThrowable1) {
      System.err.println("Could not load evaluation class for evaluation '" + str2 + "'");
      localThrowable1.printStackTrace();
      System.exit(1);
    }
    
    org.vu.contest.ContestEvaluation localContestEvaluation = null;
    try {
      localContestEvaluation = (org.vu.contest.ContestEvaluation)((Class)localContestEvaluation).newInstance();
    }
    catch (Throwable localThrowable2) {
      System.err.println("ExecutionError: Could not instantiate evaluation object for evaluation '" + str2 + "'");
      localThrowable2.printStackTrace();
      System.exit(1);
    }
    
    Class localClass = null;
    try {
      localClass = Class.forName(str1);
    }
    catch (Throwable localThrowable3) {
      System.err.println("ExecutionError: Could not load submission class for player '" + str1 + "'");
      localThrowable3.printStackTrace();
      System.exit(1);
    }
    

    if (i == 0) {
      localObject2 = new ConsertTestSecurity(null);
      System.setSecurityManager((SecurityManager)localObject2);
    }
    

    Object localObject2 = null;
    try {
      localObject2 = (ContestSubmission)localClass.newInstance();
    }
    catch (Throwable localThrowable4) {
      System.err.println("ExecutionError: Could not instantiate submission object for player '" + str1 + "'");
      localThrowable4.printStackTrace();
      System.exit(1);
    }
    

    ((ContestSubmission)localObject2).setSeed(l2);
    ((ContestSubmission)localObject2).setEvaluation(localContestEvaluation);
    

    java.util.Date localDate = new java.util.Date();
    long l3 = localDate.getTime();
    
    try
    {
      ((ContestSubmission)localObject2).run();
    } catch (SecurityException localSecurityException) {
      localSecurityException.printStackTrace();
      System.out.println("Your code has attempted a security violation!");
      System.out.println("This would terminate execution (thus no score would be assigned!)");
      System.exit(0);
    } catch (Throwable localThrowable5) {
      localThrowable5.printStackTrace();
      System.out.println("Your code has thrown an Exception/Error!");
      System.out.println("This would halt execution (the best score achieved so far would be assigned)");
    }
    

    localDate = new java.util.Date();
    long l4 = localDate.getTime() - l3;
    

    System.out.println("Score: " + Double.toString(localContestEvaluation.getFinalResult()));
    System.out.println("Runtime: " + l4 + "ms");
    
    System.exit(0);
  }
}
