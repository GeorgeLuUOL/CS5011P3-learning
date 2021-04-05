import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import org.datavec.api.transform.schema.Schema;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

public class CategoryMaker {
    Schema schema;
    RecordReaderDataSetIterator iterator;
    int size = 0;

    public CategoryMaker() {
    }

    public CategoryMaker(Schema schema) {
        this.schema = schema;
    }

    public TransformProcess.Builder buildCatagory(ArrayList<List<String>> category, HashSet<String> DropLebel) {
        TransformProcess.Builder transformProcess = new TransformProcess.Builder(schema);
        int size = schema.numColumns();
        for (int i = 0; i < size; i++) {
            String type = schema.getColumnTypes().get(i).toString();
            String name = schema.getColumnNames().get(i);
            if (type.equals("String") && !DropLebel.contains(name)) {
                transformProcess.stringToCategorical(name, category.get(i));
                transformProcess.categoricalToOneHot(name);
            }

        }
        return transformProcess;
    }

    public TransformProcess.Builder toOneHotAll(TransformProcess.Builder builder, String label) {
        int size = schema.numColumns();
        for (int i = 0; i < size; i++) {
            String type = schema.getColumnTypes().get(i).toString();
            String name = schema.getColumnNames().get(i);
            System.out.println(type + "  " + name);
            if (type.equals("String") && name != label) {
                builder.categoricalToOneHot(name);
            }

        }
        return builder;
    }


    public ArrayList<Object[]> getListFromCSV(CSVRecordReader reader) {

        ArrayList<Object[]> list = new ArrayList<>();
        while(reader.hasNext()){
            list.add(reader.next().toArray());
        }
        return list;
    }

    public String toStringInRow(ArrayList<Object[]> list, int row) {
        StringBuilder sb = new StringBuilder();
        Object[] objList=list.get(row);
        for(Object o:objList){
            sb.append(o.toString()+"  ");
        }
        return sb.toString();
    }

    public ArrayList<List<String>> makeList(CSVRecordReader reader) {
        int size = this.schema.numColumns();

        ArrayList<List<String>> fullList = new ArrayList<List<String>>(size);
        for (int i = 0; i < size; i++) {
            fullList.add(new ArrayList<String>());
        }
        while (reader.hasNext()) {
            Object[] row = reader.next().toArray();
            for (int i = 0; i < size; i++) {
                String temp = row[i].toString();
                if (!fullList.get(i).contains(temp) && schema.getColumnTypes().get(i).toString().equals("String")) {
                    fullList.get(i).add(temp);
                }
            }
        }
//        for (List<String> ss : fullList) {
//            for (String s : ss) {
//                System.out.print(s + " ");
//            }
//            System.out.println();
//        }
        return fullList;
    }

    public void showAll(CSVRecordReader reader) {
        while (reader.hasNext()) {
            System.out.println(reader.next().toArray()[3]);
        }
    }

    public void printAll() {
        while (iterator.hasNext()) {
            DataSet t = iterator.next();
            System.out.println(t.getFeatures().toStringFull());
        }
    }
}
