package com.kb.kobo.util;

import java.security.SecureRandom;
import java.util.Base64;

public class SecretKeyGenerator {

    private static final int KEY_LENGTH = 256; // 256 비트

    public static void main(String[] args) {
        // 시크릿 키 생성
        String secretKey = generateSecretKey();
        System.out.println("Generated Secret Key: " + secretKey);
    }

    public static String generateSecretKey() {
        SecureRandom secureRandom = new SecureRandom();
        byte[] key = new byte[KEY_LENGTH / 8]; // KEY_LENGTH를 바이트로 변환
        secureRandom.nextBytes(key);
        return Base64.getEncoder().encodeToString(key); // Base64 인코딩
    }
}

