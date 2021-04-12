


import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Task1 {
    Schema schema;
    RecordReaderDataSetIterator trainIterator;
    MultiLayerNetwork model;
    TransformProcess transformProcess;
    int batchSize = 80;
    DataAnalysis analysis;

    public void execute(String args[]) throws IOException, InterruptedException {
        if(args[1].equals("train")){
            System.out.println(args[1]);
            buildSchema();
            setNetwork();
            training(args[2],args[3]);
        }
        else if(args[1].equals("test")){
            System.out.println(args[1]);
            buildSchema();
            setNetwork();
            evaluation(args[2],args[3],args[4]);
        }
        else if(args[1].equals("predict")){
            System.out.println(args[1]);
            buildSchema();
            setNetwork();
            predict(args[2],args[3]);
        }
    }
    public void buildSchema() throws IOException, InterruptedException {

        schema = new Schema.Builder().addColumnsInteger("id", "amount_tsh", "gps_height")
                .addColumnsDouble("longitude", "latitude")
                .addColumnsInteger("num_private", "population")
                .addColumnCategorical("status_group", "0", "1")
                .build();


        transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("id")
                // add pre-processing for other columns here
                .build();
        Schema finalSchema = transformProcess.getFinalSchema();
        schema = finalSchema;

//        while (trainIterator.hasNext()) {
//            DataSet t = trainIterator.next();
//            System.out.println(t.toString());}
    }

    public void setNetwork() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER) //set weight initialisation method
                .activation(Activation.RELU)
                // use SGD optimiser
                .updater(new Sgd.Builder().learningRate(0.001).build())
                .list(new DenseLayer.Builder().nOut(100).build(),
                        new DenseLayer.Builder().nOut(100).build(),
// Cross-Entropy loss function
                        new OutputLayer.Builder(new LossMCXENT())
// Softmax at the output layer
                                .nOut(2).activation(Activation.SOFTMAX)
                                .build()).setInputType(InputType.feedForward(schema.numColumns() - 1)).build();
        model = new MultiLayerNetwork(config);
        model.init();


    }

    public void training(String file,String nn) throws IOException, InterruptedException {
        FileSplit inputSplit = new FileSplit(new File(file));
        int nLinesToSkip = 1; // skip the first line (header)
        TransformProcessRecordReader trainRecordReader =
                new TransformProcessRecordReader(
                        new CSVRecordReader(nLinesToSkip, ','), transformProcess);
        trainRecordReader.initialize(inputSplit);

        batchSize = 80;
        trainIterator =
                new RecordReaderDataSetIterator.Builder(trainRecordReader, batchSize)
                        // mark status_group as the output column (with 2 output classes)
                        .classification(schema.getIndexOfColumn("status_group"), 2)
                        .build();
        //ini analysis obj
        TransformProcessRecordReader recordReader = new TransformProcessRecordReader(
                new CSVRecordReader(nLinesToSkip, ','), transformProcess);
        trainRecordReader.initialize(inputSplit);
        recordReader.initialize(inputSplit);
        analysis = AnalyzeLocal.analyze(schema, recordReader);
        int nEpochs = 100;
        model.fit(trainIterator, nEpochs);

        File modelSave = new File(nn+".bin");
        model.save(modelSave);
        ModelSerializer.addObjectToFile(modelSave, "dataanalysis", analysis.toJson());
        ModelSerializer.addObjectToFile(modelSave, "schema", schema.toJson());


    }

    public void evaluation(String file,String nn,String path) throws IOException, InterruptedException {
        int nLinesToSkip = 1; // skip the first line (header)
        TransformProcessRecordReader testRecordReader =
                new TransformProcessRecordReader(
                        new CSVRecordReader(nLinesToSkip, ','), transformProcess);
        testRecordReader.initialize(new FileSplit(new File(file)));
        RecordReaderDataSetIterator testIterator =
                new RecordReaderDataSetIterator.Builder(testRecordReader, 1)
                        .classification(schema.getIndexOfColumn("status_group"), 2)
                        .build();

        Evaluation eval = new Evaluation(2);
        File modelSave = new File(nn+".bin");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);
        FileSplit inputSplit = new FileSplit(new File(file));

        CategoryMaker cm = new CategoryMaker(schema);
        int count = 0;
        CSVRecordReader reader = new CSVRecordReader(nLinesToSkip, ',');
        reader.initialize(inputSplit);
        ArrayList<Object[]> list = cm.getListFromCSV(reader);
        System.out.println("predict result:");
        StringBuilder sb = new StringBuilder();

        while (testIterator.hasNext()) {
            DataSet t = testIterator.next();
            //System.out.println(t.toString());
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            if (count < list.size()) {
                sb.append(cm.toStringInRow(list, count));
            }
            sb.append(predicted.toDoubleMatrix()[0][0]<=0.5?"functions require repair":"other");
            sb.append("\n");
            //System.out.println(labels.toStringFull());

            eval.eval(labels, predicted);
            count++;
        }
        System.out.println(sb.toString());
        TextWriter.saveAsFileWriter(sb.toString(),path);
        System.out.println(eval.stats());


    }

    public void predict(String file,String nn) throws IOException, InterruptedException {
        int nLinesToSkip = 1; // skip the first line (header)
        Schema newschema = new Schema.Builder().addColumnsInteger("id", "amount_tsh", "gps_height")
                .addColumnsDouble("longitude", "latitude")
                .addColumnsInteger("num_private", "population")
                .build();


        File modelSave = new File(nn+".bin");
        DataAnalysis analysis =
                DataAnalysis.fromJson(ModelSerializer.getObjectFromFile(
                        modelSave, "dataanalysis"));
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);
        Schema schema = Schema.fromJson(ModelSerializer.getObjectFromFile(
                modelSave, "schema"));

        transformProcess = new TransformProcess.Builder(newschema)
                .removeColumns("id")
                // add pre-processing for other columns here
                .build();
        Schema finalSchema = transformProcess.getFinalSchema();


        FileSplit inputSplit = new FileSplit(new File(file));
        CSVRecordReader raw = new CSVRecordReader(nLinesToSkip);
        raw.initialize(inputSplit);
        RecordReaderDataSetIterator rawIterator =
                new RecordReaderDataSetIterator.Builder(raw, batchSize)
                        .build();


        List<Double> idSet = new ArrayList<>();
        List<double[][]> recordSet = new ArrayList<>();
        double[][] record;
        while (rawIterator.hasNext()) {
            DataSet t = rawIterator.next();
            record = t.getFeatures().toDoubleMatrix();
            recordSet.add(record);
            //System.out.println(t.getFeatures().toStringFull());
        }
        for (double[][] matrix : recordSet) {
            for (double[] r : matrix) {
                idSet.add(r[0]);
            }
        }


        TransformProcessRecordReader predictRecordReader =
                new TransformProcessRecordReader(
                        new CSVRecordReader(nLinesToSkip, ','), transformProcess);
        predictRecordReader.initialize(inputSplit);


        RecordReaderDataSetIterator testIterator =
                new RecordReaderDataSetIterator.Builder(predictRecordReader, batchSize)
                        .build();

        //schema for predicting

        List<double[][]> resultList = new ArrayList<>();
        double[][] result;
        while (testIterator.hasNext()) {

            DataSet t = testIterator.next();
            //System.out.println(t.toString());
            INDArray features = t.getFeatures();
            //System.out.println(features.toStringFull());
            INDArray predicted = model.output(features, false);
            result = predicted.toDoubleMatrix();
            resultList.add(result);
            //System.out.println(predicted.toStringFull());
        }
        System.out.println("id required for repairing");
        int index = 0;
        for (double[][] matrix : resultList) {
            for (double[] r : matrix) {
                if (r[0] < 0.5) {
                    System.out.println(Math.round(idSet.get(index)));
                }
                index++;
            }
        }
    }
}

