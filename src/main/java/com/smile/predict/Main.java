package com.smile.predict;

import com.thoughtworks.xstream.XStream;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;

import org.apache.commons.csv.CSVFormat;
import smile.classification.LogisticRegression;
import smile.classification.LogisticRegression.Binomial;
import smile.io.Read;

public class Main {

  public static void main(String[] args)
    throws IOException, URISyntaxException {
    StringBuilder xml = new StringBuilder();
    BufferedReader reader = new BufferedReader(new FileReader("model.xml"));
    String line;
    while ((line = reader.readLine()) != null) {
      xml.append(line).append("\n");
    }

    reader.close();

    String xml2 = xml.toString();

    var xstream = new XStream();

    xstream.allowTypesByWildcard(new String[] {"smile.classification.*"});

    LogisticRegression.Binomial model = (LogisticRegression.Binomial) xstream.fromXML(
      xml2
    );

    String[] HEADERS = {
      "id",
      "reason.for.absence",
      "month.of.absence",
      "day.of.the.week",
      "seasons",
      "transportation.exp",
      "residence.dist",
      "service time",
      "age",
      "work.load",
      "hit.target",
      "discipline",
      "education",
      "son",
      "social.drinking",
      "social.smoking",
      "pet",
      "weight",
      "height",
      "bmi",
      "absent.hours",
      "absent",
    };

    CSVFormat format = CSVFormat.DEFAULT
      .builder()
      .setHeader(HEADERS)
      .setSkipHeaderRecord(true)
      .build();

    var raw_data = Read.csv("absent_split.csv", format);
    raw_data = raw_data.drop("id").drop("absent.hours").drop("absent");

    double[][] inputArray = raw_data.toArray();
    double[] probs = new double[2];

    for (int i=0; i<raw_data.nrow(); i++) {
      model.predict(inputArray[i], probs);
      System.out.println(probs[1]);
    }

    //double[] x = {27,2,4,2,179,51,18,38,251818,96,0,1,0,1,0,0,89,170,31};
    //double[] probs = new double[2];
    //model.predict(x, probs);
    //System.out.println(probs[0]);
    //System.out.println(probs[1]);
    //System.out.println(output);

    
  }
}
