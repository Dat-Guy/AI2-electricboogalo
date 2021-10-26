package net.datguy;

import net.datguy.neural.NeuralNetwork;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {

    // Referencing https://towardsdatascience.com/understanding-and-implementing-neural-networks-in-java-from-scratch-61421bb6352c

    // TODO: Network with input as ASCII, output B/W image.

    public static void main(String[] args) {

        // Import images for the skid-a-didle-a-bop (neural network)
        Path inPath = Paths.get("training");

        DirectoryStream<Path> inFolder;

        ArrayList<String> names;
        ArrayList<BufferedImage> ins;

        try {
            inFolder = Files.newDirectoryStream(inPath);
        } catch (IOException x) {
            System.err.println(x);
            return;
        }

        ins = new ArrayList<>();
        names = new ArrayList<>();

        try {
            for (Path file : inFolder) {
                ins.add(ImageIO.read(file.toFile()));
                names.add(file.toFile().getName().split("\\.")[0]);
            }
            inFolder.close();
        } catch (IOException x) {
            System.err.println(x);
            return;
        }

        // We now want to convert the image array to a
        // Training Data - 20 x 20 image with name as string input.

        double[][] fNames = new double[names.size()][];
        double[][] fIns = new double[ins.size()][];

        for (int i = 0; i < names.size(); i++) {
            fNames[i] = stringToDoubles(names.get(i), 80);
        }

        for (var i = 0; i < ins.size(); i++) {
            fIns[i] = new double[400];
            for (var r = 0; r < 20; r++) {
                for (var c = 0; c < 20; c++) {
                    fIns[i][r * 20 + c] = (ins.get(i).getRGB(c, r) >> 24) * 0.98 + 0.01;
                }
            }
        }

        // 8 - bit input (ASCII)
        // 10 character input
        // 80 neurons IN

        // 20 x 20 greyscale OUT

        NeuralNetwork nn = new NeuralNetwork(80, 500, 400);
        System.out.println("Init complete, fitting...");
        long time = System.nanoTime();
        nn.fit(fNames, fIns, 200);
        System.out.println("Completed (" + (System.nanoTime() - time) + " ns)!");
        System.out.print("Input string (max 10 chars): ");

        try {
            byte[] inTemp = new byte[10];
            System.in.read(inTemp);
            char[] inTemp2 = new char[10];

            for (int i = 0; i < 10; i++) {
                inTemp2[i] = (char) inTemp[i];
            }

            double[] input = stringToDoubles(String.copyValueOf(inTemp2), 80);
            double [] output = nn.predict(input).stream().mapToDouble(d -> d).toArray();

            BufferedImage toWrite = new BufferedImage(20, 20, BufferedImage.TYPE_4BYTE_ABGR);
            for (var r = 0; r < 20; r++) {
                for (var c = 0; c < 20; c++) {
                    char mono = (char) Math.floor(output[c + r * 20] * 255);
                    toWrite.setRGB(c, r, 0xff000000 | mono << 16 | mono << 8 | mono);
                }
            }
            System.out.println("\nImage: \n");
            for (var r = 0; r < 20; r++) {
                for (var c = 0; c < 20; c++) {
                    System.out.print(output[c + r * 20] > 0.5 ? "#" : " ");
                }
                System.out.println(' ');
            }
            File outTemp = new File("result/" + String.valueOf(inTemp2) + ".png");
            ImageIO.write(toWrite, "png", outTemp);

        } catch (IOException x) {
            System.err.println(x);
        }
    }

    public static double[] stringToDoubles(String s, int length) {

        if (length % 8 != 0) {
            return null;
        }

        double[] out = new double[length];

        if (s.isEmpty() || s.isBlank()) {
            return out;
        }

        for (int i = 0; i < Math.min(length, s.length()); i+=8) {
            out[i] = s.charAt(i) & 0b10000000;
            out[i+1] = s.charAt(i) & 0b1000000;
            out[i+2] = s.charAt(i) & 0b100000;
            out[i+3] = s.charAt(i) & 0b10000;
            out[i+4] = s.charAt(i) & 0b1000;
            out[i+5] = s.charAt(i) & 0b100;
            out[i+6] = s.charAt(i) & 0b10;
            out[i+7] = s.charAt(i) & 0b1;
        }

        return out;
    }
}