package com.kb.kobo.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.kb.kobo.service.OpenAiService;
import com.kb.kobo.entity.Program;
import java.util.List;


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
        List<Program> programs = openAiService.callOpenAiApi(input);
        return ResponseEntity.ok(programs);
    }
}
