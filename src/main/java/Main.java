/*
 * A (slightly) modified example of using dl4j for classification. 
 * The original example can be downloaded from https://github.com/treo/quickstart-with-dl4j
 * A blog post explaining in details all steps can be found at: https://www.dubs.tech/guides/quickstart-with-dl4j/
 * For more examples on how to use dl4j, see: 
 *     The official website of dl4j: https://deeplearning4j.konduit.ai/
 *     dl4j blog: https://blog.konduit.ai/
 *     dl4j-examples repository: https://github.com/eclipse/deeplearning4j-examples
 *     dl4j community forum, where you can post questions and get support: https://community.konduit.ai/
 */

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;


public class Main {

    public static void splitTrainAndTest(String inputPathname) throws IOException {
    	System.out.println("");
    	System.out.println("----------------- Split data into train and test sets ---------");
        double splitAt = 0.8;

         File inputFile = new File(inputPathname);
         List<String> lines = Files.readAllLines(inputFile.toPath());
         String header = lines.get(0);
         lines.remove(0);

         int splitPosition = (int)Math.round(splitAt * lines.size());


         Random random = new Random();
        random.setSeed(0xC0FFEE);

        Collections.shuffle(lines, random);

        ArrayList<String> train = new ArrayList<>(lines.subList(0, splitPosition));
        ArrayList<String> test = new ArrayList<>(lines.subList(splitPosition, lines.size()));
        train.add(0, header);
    	test.add(0, header);

        String outputTrainPathname = inputPathname.replaceAll(".csv", "-train.csv");
    	String outputTestPathname = inputPathname.replaceAll(".csv", "-test.csv");
    	Files.write(Paths.get(outputTrainPathname), train);
    	Files.write(Paths.get(outputTestPathname), test);               
    }

    
    public static void Analysis(String inputPathname) throws Exception {
    	System.out.println("");
    	System.out.println("----------------- Analyse ---------");
    	
    	Random random = new Random();
        random.setSeed(0xC0FFEE);
        FileSplit inputSplit = new FileSplit(new File(inputPathname));
        
        int nLinesToSkip = 1; // skip the first line (header)
        CSVRecordReader recordReader = new CSVRecordReader(nLinesToSkip,',');
        recordReader.initialize(inputSplit);

        Schema schema = new Schema.Builder()
                .addColumnsInteger("Row Number", "Customer Id")
                .addColumnString("Surname")
                .addColumnInteger("Credit Score")
                .addColumnCategorical("Geography", "France", "Germany", "Spain")
                .addColumnCategorical("Gender", "Female", "Male")
                .addColumnsInteger("Age", "Tenure")
                .addColumnDouble("Balance")
                .addColumnInteger("Num Of Products")
                .addColumnCategorical("Has Credit Card", "0", "1")
                .addColumnCategorical("Is Active Member", "0", "1")
                .addColumnDouble("Estimated Salary")
                .addColumnCategorical("Exited", "0", "1")
                .build();

        DataAnalysis analysis = AnalyzeLocal.analyze(schema, recordReader);
        HtmlAnalysis.createHtmlAnalysisFile(analysis, new File(inputPathname.replaceAll(".csv", "-analyse.html")));        
    }
    
   
    
    public static void main(String[] args) throws Exception {
//       Task1 task1=new Task1();
//        task1.buildSchema();
//        task1.setNetwork();
//        task1.training();
//        task1.evaluation();
//        task1.predict();

        Task2 task2=new Task2();
        task2.buildSchema();
        task2.setNetwork();
        task2.training();
        task2.evaluation();
        task2.predict();
    }
}
