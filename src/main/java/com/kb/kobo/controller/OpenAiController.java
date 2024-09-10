package com.kb.kobo.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.kb.kobo.service.OpenAiService;
import com.kb.kobo.model.Program;

import java.util.List; // List를 사용하기 위해 추가
import java.util.ArrayList; // ArrayList를 사용하기 위해 추가

@RestController
@RequestMapping("/api")
public class OpenAiController {

    private final OpenAiService openAiService;

    @Autowired
    public OpenAiController(OpenAiService openAiService) {
        this.openAiService = openAiService;
    }

    @GetMapping("/programs")
    public ResponseEntity<List<Program>> getPrograms() {
        String input = "Provide data for programs.";
        List<Program> programs = openAiService.callOpenAiApi(input); // 인자를 전달
        return ResponseEntity.ok(programs);
    }
}
