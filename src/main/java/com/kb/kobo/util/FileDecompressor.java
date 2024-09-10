package com.kb.kobo.util;

import java.io.*;
import java.util.Base64;
import java.util.zip.InflaterInputStream;

public class FileDecompressor {
    public static void decodeAndDecompressFile(File base64File, File outputFile) throws IOException {
        String base64EncodedData;
        try (BufferedReader reader = new BufferedReader(new FileReader(base64File))) {
            base64EncodedData = reader.readLine();
        }

        byte[] compressedBytes = Base64.getDecoder().decode(base64EncodedData);

        try (ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(compressedBytes);
             InflaterInputStream inflaterInputStream = new InflaterInputStream(byteArrayInputStream);
             FileOutputStream fileOutputStream = new FileOutputStream(outputFile)) {

            byte[] buffer = new byte[1024];
            int length;
            while ((length = inflaterInputStream.read(buffer)) > 0) {
                fileOutputStream.write(buffer, 0, length);
            }
        }
    }

    public static void main(String[] args) {
        try {
            File compressedFile = new File("src/main/resources/uploads/HelloWorldCompressed.b64");
            File outputFile = new File("src/main/resources/uploads/DecompressedHelloWorld.pdf");
            decodeAndDecompressFile(compressedFile, outputFile);
            System.out.println("파일 복원 성공");
        } catch (IOException e) {
            System.err.println("파일 복원 오류 발생: " + e.getMessage());
        }
    }
}


