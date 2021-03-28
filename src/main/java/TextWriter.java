import java.io.FileWriter;
import java.io.IOException;

public class TextWriter{
        public static String filePath = "testResult.txt";

        public static void saveAsFileWriter(String content) {
            FileWriter fwriter = null;
            try {
                //do not over write
                fwriter = new FileWriter(filePath);
                fwriter.write(content);
            } catch (IOException ex) {
                ex.printStackTrace();
            } finally {
                try {
                    fwriter.flush();
                    fwriter.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        }
}
