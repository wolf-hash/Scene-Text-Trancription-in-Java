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
        tesseract.setTessVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
        tesseract.setPageSegMode(6);
        tesseract.setOcrEngineMode(2);
        tesseract.setTessVariable("user_defined_dpi", "300");

        try {
            String result = tesseract.doOCR(image);
            System.out.print(result);
        } catch (TesseractException e) {
            e.printStackTrace();
        }

    }
}

