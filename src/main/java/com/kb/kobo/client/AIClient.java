package com.kb.kobo.client;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import org.json.JSONObject;

public class AIClient {

    public String analyzeQuestion(Long documentId, String content) throws Exception {
        JSONObject json = new JSONObject();
        json.put("documentId", documentId);
        json.put("content", content);

        // HttpClient 생성
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMinutes(30))
                .build();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(new URI("http://localhost:8080/"))  // AI 서버의 분석 요청 URL
                .header("Content-Type", "application/json")
                .timeout(Duration.ofMinutes(30)) // 응답 대기 30분 타임아웃
                .POST(HttpRequest.BodyPublishers.ofString(json.toString()))
                .build();

        // 요청 보내기 및 응답 받기
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            return response.body();  // AI 서버로부터 받은 결과
        } else {
            throw new RuntimeException("AI 서버 호출 실패: " + response.statusCode());
        }
    }
}
