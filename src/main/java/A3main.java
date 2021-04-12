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


public class A3main {


    public static void main(String[] args) throws Exception {
     if(args.length<2){
      System.out.println("Usage:  <task_id> <train|test|predict> <other_arguments>");
      return ;
     }

     switch (args[0]){
      case "task1":
       System.out.println(args[0]);
       Task1 t1=new Task1();
       t1.execute(args);
       break;
      case "task2":
       System.out.println(args[0]);
       Task2 t2=new Task2();
       t2.execute(args);
       break;
      case"task3":
       System.out.println(args[0]);
       Task3 t3=new Task3();
       t3.execute(args);
       break;
      case "task4":
       System.out.println(args[0]);
       Task4 t4=new Task4();
       t4.execute(args);
       break;
       default:
        System.out.println("test case does not exit");
        break;

     }
//       Task1 task1=new Task1();
//        task1.buildSchema();
//        task1.setNetwork();
//        //task1.training();
//        task1.evaluation();
//        task1.predict();

//        Task2 task2=new Task2();
//        task2.buildSchema();
//        task2.setNetwork();
//        task2.training();
//        task2.evaluation();
//        task2.predict();
//
//        Task3 task3=new Task3();
//        task3.buildSchema();
//        task3.setNetwork();
//        task3.training();
//        task3.evaluation();
//        task3.predict();
//
//        Task4 task4=new Task4();
//        task4.buildSchema();
//        task4.setNetwork();
//        task4.training();
//        task4.evaluation();
//        task4.predict();
    }
}
