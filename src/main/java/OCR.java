import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import org.opencv.core.Core;
import org.opencv.core.*;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.*;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.*;

public class OCR {

  public static void main(String[] args) {
    nu.pattern.OpenCV.loadLocally();

    float scoreThresh = 0.3f;
    float nmsThresh = 0.1f;
    BufferedImage image = null;
    try {
      image = ImageIO.read(new File("src/main/resources/TestImages/test2.jpg"));
    } catch (IOException e) {
      e.printStackTrace();
    }

    Mat frame = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
    byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
    frame.put(0,0, data);

    Net net = Dnn.readNetFromTensorflow("src/main/resources/frozen_east_text_detection.pb");
    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

    Size siz = new Size(320, 320);
    int W = (int)(siz.width / 4); // width of the output geometry  / score maps
    int H = (int)(siz.height / 4); // height of those. the geometry has 4, vertically stacked maps, the score one 1
    Mat blob = Dnn.blobFromImage(frame, 1.0,siz, new Scalar(123.68, 116.78, 103.94), true, false);
    net.setInput(blob);
    List<Mat> outs = new ArrayList<>(2);
    List<String> outNames = new ArrayList<String>();
    outNames.add("feature_fusion/Conv_7/Sigmoid");
    outNames.add("feature_fusion/concat_3");
    net.forward(outs, outNames);

    // Decode predicted bounding boxes.
    Mat scores = outs.get(0).reshape(1, H);
    Mat geometry = outs.get(1).reshape(1, 5 * H);
    List<Float> confidencesList = new ArrayList<>();
    List<RotatedRect> boxesList = decode(scores, geometry, confidencesList, scoreThresh);

    String text = "";

    MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confidencesList));
    RotatedRect[] boxesArray = boxesList.toArray(new RotatedRect[0]);
    MatOfRotatedRect boxes = new MatOfRotatedRect(boxesArray);
    MatOfInt indices = new MatOfInt();
    Dnn.NMSBoxesRotated(boxes, confidences, scoreThresh, nmsThresh, indices);
    // Render detections
    Point ratio = new Point((float)frame.cols()/siz.width, (float)frame.rows()/siz.height);
    int[] indexes = indices.toArray();
    for(int i = 0; i<indexes.length; ++i) {
      RotatedRect rot = boxesArray[indexes[i]];
      Point[] vertices = new Point[4];
      rot.points(vertices);
      for (int j = 0; j < 4; ++j) {
        vertices[j].x *= ratio.x;
        vertices[j].y *= ratio.y;
      }

      double min_x = vertices[0].x, min_y = vertices[0].y;
      double max_x = 0, max_y = 0;
      for (int j = 0; j < 4; j++) {
        if(vertices[j].x < min_x){
          min_x = vertices[j].x;
        }
        if(vertices[j].y < min_y){
          min_y = vertices[j].y;
        }
        if(vertices[j].x > max_x){
          max_x = vertices[j].x;
        }
        if(vertices[j].y > max_y){
          max_y = vertices[j].y;
        }
      }

      TesseractRecognizer tess4j = new TesseractRecognizer();


      Point tl = new Point(vertices[1].x - 5, vertices[1].y - 5);
      Point tr = new Point(vertices[2].x + 5, vertices[2].y - 5);
      Point br = new Point(vertices[3].x + 5, vertices[3].y + 5);
      Point bl = new Point(vertices[0].x - 5, vertices[0].y + 5);

      Mat destImage = new Mat((int)(max_y - min_y), (int)(max_x - min_x)  , frame.type());
      Mat src = new MatOfPoint2f(tl, tr, br, bl);
      Mat dst = new MatOfPoint2f(new Point(0, 0), new Point(destImage.width() - 1, 0), new Point(destImage.width() - 1, destImage.height() - 1), new Point(0, destImage.height() - 1));
      Mat transform = Imgproc.getPerspectiveTransform(src, dst);
      Imgproc.warpPerspective(frame, destImage, transform, destImage.size());
      Imgcodecs.imwrite("src/main/resources/Debugging/warped.jpg", destImage);


      Imgproc.cvtColor(destImage, destImage, Imgproc.COLOR_BGR2GRAY);
      Imgcodecs.imwrite("src/main/resources/Debugging/gray.jpg", destImage);
      Imgproc.GaussianBlur(destImage, destImage, new Size(3, 3),0);
//
//      Imgproc.adaptiveThreshold(destImage, destImage, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 55, 4 );
      Imgproc.threshold(destImage, destImage, 80, 255, Imgproc.THRESH_BINARY);
      if(destImage.get(0,0)[0] == 0){
        Core.bitwise_not(destImage, destImage);
      }
      Imgproc.GaussianBlur(destImage, destImage, new Size(3, 3),0);

      Size scaleSize = new Size(destImage.width() * 2, destImage.height() * 2);
      Imgproc.resize(destImage, destImage, scaleSize, 0, 0, Imgproc.INTER_CUBIC);
      Imgproc.GaussianBlur(destImage, destImage, new Size(3, 3),0);


      Imgcodecs.imwrite("src/main/resources/Debugging/testing.jpg", destImage);


      MatOfByte matOfByte = new MatOfByte();
      Imgcodecs.imencode(".jpg", destImage, matOfByte);
      byte[] byteArray = matOfByte.toArray();
      InputStream in = new ByteArrayInputStream(byteArray);
      try {
        BufferedImage bufImage = ImageIO.read(in);
        text = text + " " + tess4j.recognizer(bufImage);

      } catch (IOException e) {
        e.printStackTrace();
      }

      for (int j = 0; j < 4; ++j) {
        Imgproc.line(frame, vertices[j], vertices[(j + 1) % 4], new Scalar(0, 0,255), 1);
      }
//break;
    }
    Imgcodecs.imwrite("src/main/resources/Debugging/out.jpg", frame);
    System.out.println(text);
  }

  private static List<RotatedRect> decode(Mat scores, Mat geometry, List<Float> confidences, float scoreThresh) {
    // size of 1 geometry plane
    int W = geometry.cols();
    int H = geometry.rows() / 5;

    List<RotatedRect> detections = new ArrayList<>();
    for (int y = 0; y < H; ++y) {
      Mat scoresData = scores.row(y);
      Mat x0Data = geometry.submat(0, H, 0, W).row(y);
      Mat x1Data = geometry.submat(H, 2 * H, 0, W).row(y);
      Mat x2Data = geometry.submat(2 * H, 3 * H, 0, W).row(y);
      Mat x3Data = geometry.submat(3 * H, 4 * H, 0, W).row(y);
      Mat anglesData = geometry.submat(4 * H, 5 * H, 0, W).row(y);

      for (int x = 0; x < W; ++x) {
        double score = scoresData.get(0, x)[0];
        if (score >= scoreThresh) {
          double offsetX = x * 4.0;
          double offsetY = y * 4.0;
          double angle = anglesData.get(0, x)[0];
          double cosA = Math.cos(angle);
          double sinA = Math.sin(angle);
          double x0 = x0Data.get(0, x)[0];
          double x1 = x1Data.get(0, x)[0];
          double x2 = x2Data.get(0, x)[0];
          double x3 = x3Data.get(0, x)[0];
          double h = x0 + x2;
          double w = x1 + x3;
          Point offset = new Point(offsetX + cosA * x1 + sinA * x2, offsetY - sinA * x1 + cosA * x2);
          Point p1 = new Point(-1 * sinA * h + offset.x, -1 * cosA * h + offset.y);
          Point p3 = new Point(-1 * cosA * w + offset.x,      sinA * w + offset.y);
          RotatedRect r = new RotatedRect(new Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y)), new Size(w, h), -1 * angle * 180 / Math.PI);
          detections.add(r);
          confidences.add((float) score);
        }
      }
    }
    return detections;
  }

}

class TesseractRecognizer {

  public String recognizer(BufferedImage image){
    Tesseract tesseract = new Tesseract();
    tesseract.setDatapath("src/main/resources/tessdata");
    tesseract.setLanguage("eng");
    tesseract.setTessVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
    tesseract.setPageSegMode(6);
    tesseract.setOcrEngineMode(2);
    tesseract.setTessVariable("user_defined_dpi", "300");


    try {
      String result = tesseract.doOCR(image);
      return result;
    } catch (TesseractException e) {
      e.printStackTrace();
      return null;
    }
  }
}