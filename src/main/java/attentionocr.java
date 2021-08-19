import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import javax.imageio.ImageIO;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import java.nio.file.Paths;

import java.nio.file.Files;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.charset.Charset;

public class attentionocr {
    private static String imagepath = "src/main/resources/testocr.png";
    private static String modelpath = "src/main/resources/aocr_frozen_graph/";
    private static byte[] graphDef;
    private static byte[] imageBytes;

    private static byte[] readAllBytesOrExit(Path path)  {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;

    }


    private static String executeAOCRGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("input_image_as_bytes", image).fetch("prediction").run().get(0)) {

                String rstring = new String(result.bytesValue(), Charset.forName("UTF-8"));


                return rstring;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("TensorFlow Version" + TensorFlow.version());

        graphDef = readAllBytesOrExit(Paths.get(modelpath, "frozen_model_temp.pb"));

        BufferedImage img = null;
        try {
            img = ImageIO.read(new File("src/main/resources/test.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(img, "jpg", baos);
        imageBytes = baos.toByteArray();


        try (Tensor image = Tensor.create(imageBytes)) {
            String result = executeAOCRGraph(graphDef, image);

            System.out.println(result);

        }
    }
}
