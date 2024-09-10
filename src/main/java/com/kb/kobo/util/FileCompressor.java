package com.kb.kobo.util;

import java.io.*;
import java.util.Base64;
import java.util.zip.DeflaterOutputStream;

public class FileCompressor {
    public static void compressAndEncodeFile(File file, File outputFile) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        try (FileInputStream fileInputStream = new FileInputStream(file);
             DeflaterOutputStream deflaterOutputStream = new DeflaterOutputStream(byteArrayOutputStream)) {

            byte[] buffer = new byte[1024];
            int length;
            while ((length = fileInputStream.read(buffer)) > 0) {
                deflaterOutputStream.write(buffer, 0, length);
            }
        }
        byte[] compressedBytes = byteArrayOutputStream.toByteArray();
        String encodedString = Base64.getEncoder().encodeToString(compressedBytes);
        try (FileWriter fileWriter = new FileWriter(outputFile)) {
            fileWriter.write(encodedString);
        }
    }

    public static void main(String[] args) {
        try {
            File file = new File("src/main/resources/uploads/HelloWorld.pdf");
            File compressedFile = new File("src/main/resources/uploads/HelloWorldCompressed.b64");
            compressAndEncodeFile(file, compressedFile);
            System.out.println("압축된 Base64 파일 생성 완료: " + compressedFile.getAbsolutePath());
        } catch (IOException e) {
            System.err.println("파일 압축 및 인코딩 오류 발생: " + e.getMessage());
        }
    }
}
