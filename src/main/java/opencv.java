import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import net.sourceforge.tess4j.util.ImageHelper;
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

public class opencv {


  public static void main(String[] args) {
    nu.pattern.OpenCV.loadLocally();

    float scoreThresh = 0.5f;
    float nmsThresh = 0.1f;
//////////////////////////////////
    BufferedImage image = null;
    try {
      image = ImageIO.read(new File("src/main/resources/test3.jpg"));
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
    for(int i = 0; i<indexes.length;++i) {
      RotatedRect rot = boxesArray[indexes[i]];
      Point[] vertices = new Point[4];
      rot.points(vertices);
      for (int j = 0; j < 4; ++j) {
        vertices[j].x *= ratio.x;
        vertices[j].y *= ratio.y;
      }
      GeneralPath clip = new GeneralPath();
      clip.moveTo(vertices[0].x - 10, vertices[0].y + 10);
      clip.lineTo(vertices[1].x - 10, vertices[1].y - 10);
      clip.lineTo(vertices[2].x + 10, vertices[2].y - 10);
      clip.lineTo(vertices[3].x + 10, vertices[3].y + 10);
      clip.closePath();

      double min_x = vertices[0].x, min_y = vertices[0].y;
      for (int j = 0; j < 4; j++) {
        if(vertices[j].x < min_x){
          min_x = vertices[j].x;
        }
        if(vertices[j].y < min_y){
          min_y = vertices[j].y;
        }
      }
      Rectangle rect = clip.getBounds();
      BufferedImage img = new BufferedImage(rect.width, rect.height, BufferedImage.TYPE_3BYTE_BGR);
      Graphics2D g2d = img.createGraphics();
      clip.transform(AffineTransform.getTranslateInstance(-min_x, -min_y));
      g2d.setClip(clip);
      g2d.translate(-min_x, -min_y);

      g2d.drawImage(image, 0, 0, null);
      g2d.dispose();

      try {
        ImageIO.write(img, "png", new File("src/main/resources/Clipped.png"));
      } catch (IOException e) {
        e.printStackTrace();
      }

      TesseractRecognizer tess4j = new TesseractRecognizer();

      Mat p_img = new Mat(img.getHeight(), img.getWidth(), CvType.CV_8UC3);
      byte[] p_data = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
      p_img.put(0,0, p_data);
      Imgproc.cvtColor(p_img, p_img, Imgproc.COLOR_BGR2GRAY);
      Imgproc.GaussianBlur(p_img, p_img, new Size(3, 3),0);
      Imgproc.adaptiveThreshold(p_img, p_img, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C ,Imgproc.THRESH_BINARY_INV, 99, 4);

      Imgcodecs.imwrite("src/main/resources/testing.jpg", p_img);


      MatOfByte matOfByte = new MatOfByte();
      Imgcodecs.imencode(".jpg", p_img, matOfByte);
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
//      break;
    }
    Imgcodecs.imwrite("src/main/resources/out.jpg", frame);
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
    tesseract.setPageSegMode(8);
    tesseract.setOcrEngineMode(2);
    try {
      String result = tesseract.doOCR(image);
      return result;
    } catch (TesseractException e) {
      e.printStackTrace();
      return null;
    }
  }
}