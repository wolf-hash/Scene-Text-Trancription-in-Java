import java.awt.image.BufferedImage;
import java.io.IOException;
import javax.imageio.ImageIO;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;

import java.io.File;

public class tesseract {
    public static void main(String[] args) {
        BufferedImage image = null;
        try {
            image = ImageIO.read(new File("src/main/resources/Clipped.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        Tesseract tesseract = new Tesseract();
        tesseract.setDatapath("src/main/resources/tessdata");
        tesseract.setLanguage("eng");
        tesseract.setPageSegMode(1);
        tesseract.setOcrEngineMode(1);
        try {
            String result = tesseract.doOCR(image);
            System.out.print(result);
        } catch (TesseractException e) {
            e.printStackTrace();
        }

    }
}


    ////////////////////////////////////////////////////////////////////////////////////////
//    Tesseract tesseract = new Tesseract();
//      tesseract.setDatapath("src/main/resources/tessdata");
//          tesseract.setLanguage("eng");
//          tesseract.setPageSegMode(1);
//          tesseract.setOcrEngineMode(1);
//          try {
//          String result = tesseract.doOCR(image, new Rectangle((int)vertices[1].x, (int)vertices[1].y , (int)(vertices[2].x-vertices[0].x), (int)(vertices[0].y-vertices[2].y)));
//          System.out.print(result);
//          } catch (TesseractException e) {
//          e.printStackTrace();
//          }
/////////////////////////////////////////////////////////////////////////////////////////