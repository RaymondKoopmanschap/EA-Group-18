import java.io.FileDescriptor;
import java.net.InetAddress;
import java.security.Permission;
import java.util.Vector;

public class ConsertTestSecurity
  extends SecurityManager
{
  private Vector<String> permitted_reads_;
  private String target_;
  
  public ConsertTestSecurity(String paramString)
  {
    target_ = paramString;
    permitted_reads_ = new Vector();
  }
  
  public void addPermittedRead(String paramString)
  {
    permitted_reads_.add(paramString);
  }
  


  public void checkRead(String paramString)
  {
    //if (!permitted_reads_.contains(paramString)) throw new SecurityException("Attempting to read file! (" + paramString + ")");
  }
  
  public void checkRead(String paramString, Object paramObject) {
    //if (!permitted_reads_.contains(paramString)) throw new SecurityException("Attempting to read file! (" + paramString + ")");
  }
  
  public void checkRead(FileDescriptor paramFileDescriptor) {
    //throw new SecurityException("Attempting to read file! (" + paramFileDescriptor.toString() + ")");
  }
  
  public void checkWrite(String paramString)
  {
    //throw new SecurityException("Attempting to write file! (" + paramString + ")");
  }
  
  public void checkWrite(FileDescriptor paramFileDescriptor) {
    //throw new SecurityException("Attempting to write file! (" + paramFileDescriptor.toString() + ")");
  }
  
  public void checkDelete(String paramString)
  {
    //throw new SecurityException("Attempting to delete file! (" + paramString + ")");
  }
  


  public void checkAccept(String paramString, int paramInt)
  {
    throw new SecurityException("Attempting to accept a connection! (" + paramString + ":" + paramInt + ")");
  }
  
  public void checkConnect(String paramString, int paramInt) {
    throw new SecurityException("Attempting to start a connection! (" + paramString + ":" + paramInt + ")");
  }
  
  public void checkConnect(String paramString, int paramInt, Object paramObject) {
    throw new SecurityException("Attempting to start a connection! (" + paramString + ":" + paramInt + ")");
  }
  
  public void checkListen(int paramInt) {
    throw new SecurityException("Attempting to listen to a port! (" + paramInt + ")");
  }
  
  public void checkMulticast(InetAddress paramInetAddress) {
    throw new SecurityException("Attempting to use an IP multicast! (" + paramInetAddress.getCanonicalHostName() + ")");
  }
  
  public void checkSetFactory() {
    throw new SecurityException("Attempting to access the socket factory!");
  }
  


  public void checkExec(String paramString)
  {
    throw new SecurityException("Attempting to create a new process! ('" + paramString + "')");
  }
  
  public void checkLink(String paramString) {
    throw new SecurityException("Attempting to load a binary! ('" + paramString + "')");
  }
  
  public void checkPropertiesAccess() {
    throw new SecurityException("Attempting to access system properties!");
  }
  



  public void checkPrintJobAccess()
  {
    throw new SecurityException("Attempting to request a print job!");
  }
  
  public void checkCreateClassLoader() {
    throw new SecurityException("Attempting to create a class loader!");
  }
  
  public void checkPermission(Permission paramPermission)
  {
    if (paramPermission.getActions().contains("write")) throw new SecurityException("Requesting permission to write system property! (" + paramPermission.getName() + ")");
  }
}
