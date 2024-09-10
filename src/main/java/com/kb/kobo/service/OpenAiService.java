package com.kb.kobo.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.kb.kobo.model.Program;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;

import java.util.List;

@Service
public class OpenAiService {

    @Value("${openai.api.key}")
    private String apiKey;

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public OpenAiService(RestTemplate restTemplate, ObjectMapper objectMapper) {
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
    }

    public List<Program> callOpenAiApi(String input) {
        String apiUrl = "https://api.odcloud.kr/api/3034791/v1/uddi:80a74cfd-55d2-4dd3-81c7-d01567d0b3c4?page=1&perPage=100&serviceKey=dq0FiphIXegKyP%2F5zIDul95IvtalzdhixfdY7Hp9g4Onm%2FX9aCt378S5nVejoKY%2BGFEL5uvq75P1%2FlYuu%2Bf%2BLQ%3D%3D";

        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer " + apiKey);
        headers.set("Content-Type", "application/json");

        String requestBody = "{\"model\": \"gpt-3.5-turbo\", \"messages\": [{\"role\": \"user\", \"content\": \""
                + input + "\"}], \"max_tokens\": 100}";

        HttpEntity<String> entity = new HttpEntity<>(requestBody, headers);

        ResponseEntity<String> response = restTemplate.exchange(
                apiUrl,
                HttpMethod.POST,
                entity,
                String.class);

        // JSON 응답을 List<Program>으로 변환
        try {
            // 응답을 파싱하여 List<Program>으로 변환
            List<Program> programs = objectMapper.readValue(response.getBody(),
                    objectMapper.getTypeFactory().constructCollectionType(List.class, Program.class));
            return programs;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
