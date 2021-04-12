import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import java.util.HashSet;
import java.util.List;

public class Task3 {
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
            training();
        }
        else if(args[1].equals("test")){
            System.out.println(args[1]);
            buildSchema();
            setNetwork();
            evaluation();
        }
        else if(args[1].equals("predict")){
            System.out.println(args[1]);
            buildSchema();
            setNetwork();
            predict();
        }
    }
    public void buildSchema() throws IOException, InterruptedException {

        schema = new Schema.Builder().addColumnsInteger("id", "amount_tsh")
                .addColumnString("date_recorded")
                .addColumnString("funder")
                .addColumnsInteger("gps_height")
                .addColumnsString("installer")
                .addColumnsDouble("longitude", "latitude")
                .addColumnsString("wpt_name")
                .addColumnsInteger("num_private")
                .addColumnsString("basin", "subvillage", "region")
                .addColumnsInteger("region_code", "district_code")
                .addColumnsString("lga", "ward")
                .addColumnsInteger("population")
                .addColumnString("public_meeting")
                .addColumnsString("recorded_by", "scheme_management", "scheme_name")
                .addColumnString("permit")
                .addColumnsInteger("construction_year")
                .addColumnsString("extraction_type", "extraction_type_group", "extraction_type_class", "management", "management_group"
                        , "payment", "payment_type", "water_quality", "quality_group", "quantity", "quantity_group", "source", "source_type",
                        "source_class", "waterpoint_type", "waterpoint_type_group")
                .addColumnCategorical("status_group", "0", "1")
                .build();


        //System.out.println(schema.numColumns());
        int nLinesToSkip = 1; // skip the first line (header)
        FileSplit inputSplit = new FileSplit(new File("task3_train.csv"));

        CSVRecordReader reader = new CSVRecordReader(nLinesToSkip, ',');
        reader.initialize(inputSplit);
        CategoryMaker cm = new CategoryMaker(schema);
        ArrayList<List<String>> fullList = cm.makeList(reader);


        HashSet<String> dropLabel = new HashSet<>();
        dropLabel.add("id");
        dropLabel.add("date_recorded");
        dropLabel.add("ward");
        dropLabel.add("lga");
        dropLabel.add("basin");
        dropLabel.add("subvillage");
        dropLabel.add("region");
        dropLabel.add("wpt_name");
        dropLabel.add("scheme_name");
        dropLabel.add("installer");
        dropLabel.add("funder");


        TransformProcess.Builder builder = cm.buildCatagory(fullList, dropLabel)
                .removeColumns("id")
                .removeColumns("date_recorded")
                .removeColumns("funder")
                .removeColumns("installer")
                .removeColumns("ward")
                .removeColumns("lga")
                .removeColumns("basin")
                .removeColumns("subvillage")
                .removeColumns("region")
                .removeColumns("wpt_name")
                .removeColumns("scheme_name");



        transformProcess = builder.build();

        Schema tempSchema = transformProcess.getFinalSchema();
        schema = tempSchema;
        cm.schema = schema;

        //System.out.println(schema.toString());


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

        //System.out.println(schema.toString());
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

    public void training() throws IOException {
        int nEpochs = 100;
        model.fit(trainIterator, nEpochs);
        File modelSave = new File("task3_train-model.bin");
        model.save(modelSave);
        //ModelSerializer.addObjectToFile(modelSave, "dataanalysis", analysis.toJson());
        //ModelSerializer.addObjectToFile(modelSave, "schema", schema.toJson());


    }

    public void evaluation() throws IOException, InterruptedException {
        int nLinesToSkip = 1; // skip the first line (header)
        TransformProcessRecordReader testRecordReader =
                new TransformProcessRecordReader(
                        new CSVRecordReader(nLinesToSkip, ','), transformProcess);
        testRecordReader.initialize(new FileSplit(new File("task3_test.csv")));
        RecordReaderDataSetIterator testIterator =
                new RecordReaderDataSetIterator.Builder(testRecordReader, 1)
                        .classification(schema.getIndexOfColumn("status_group"), 2)
                        .build();

        Evaluation eval = new Evaluation(2);
        File modelSave = new File("task3_train-model.bin");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);
        FileSplit inputSplit = new FileSplit(new File("task3_train.csv"));

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
        TextWriter.saveAsFileWriter(sb.toString(),"result-task3.txt");


// print accuracy
        //System.out.println("Accuracy: " + eval.accuracy());
        System.out.println(eval.stats());
    }

    public void predict() throws IOException, InterruptedException {
        int nLinesToSkip = 1; // skip the first line (header)
        TransformProcessRecordReader testRecordReader =
                new TransformProcessRecordReader(
                        new CSVRecordReader(nLinesToSkip, ','), transformProcess);
        testRecordReader.initialize(new FileSplit(new File("task3_test.csv")));
        RecordReaderDataSetIterator testIterator =
                new RecordReaderDataSetIterator.Builder(testRecordReader, 1)
                        .classification(schema.getIndexOfColumn("status_group"), 2)
                        .build();

        Evaluation eval = new Evaluation(2);
        File modelSave = new File("task3_train-model.bin");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);
        FileSplit inputSplit = new FileSplit(new File("task3_test.csv"));

        CategoryMaker cm = new CategoryMaker(schema);
        int count = 0;
        CSVRecordReader reader = new CSVRecordReader(nLinesToSkip, ',');
        reader.initialize(inputSplit);
        ArrayList<Object[]> list = cm.getListFromCSV(reader);
        System.out.println("id requires repair");
        StringBuilder sb = new StringBuilder();

        while (testIterator.hasNext()) {
            DataSet t = testIterator.next();
            //System.out.println(t.toString());
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            if (count < list.size() && predicted.toDoubleMatrix()[0][0] <= 0.5) {
                sb.append(cm.toStringInRow(list, count).split(" ")[0]);
                sb.append("\n");
            }

            //System.out.println(labels.toStringFull());

            eval.eval(labels, predicted);
            count++;
        }
        System.out.println(sb.toString());

    }
}
