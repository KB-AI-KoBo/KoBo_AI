package com.kb.kobo.controller;

import com.kb.kobo.entity.SupportProgram;
import com.kb.kobo.service.SupportProgramService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.sql.Date;
import java.util.List;

@RestController
@CrossOrigin(origins = "http://localhost:3000")
@RequestMapping("/api/support-programs")
public class SupportProgramController {

    private final SupportProgramService supportProgramService;

    @Autowired
    public SupportProgramController(SupportProgramService supportProgramService) {
        this.supportProgramService = supportProgramService;
    }

    @PostMapping
    public ResponseEntity<SupportProgram> createSupportProgram(
            @RequestParam String programName,
            @RequestParam String url){
//            @RequestParam String description,
//            @RequestParam String eligibilityCriteria,
//            @RequestParam String applicationDeadline) {


        // SupportProgram 객체 생성
        SupportProgram supportProgram = new SupportProgram();
        supportProgram.setProgramName(programName);
        supportProgram.setUrl(url);
//        supportProgram.setDescription(description);
//        supportProgram.setEligibilityCriteria(eligibilityCriteria);
//        supportProgram.setApplicationDeadline(Date.valueOf(applicationDeadline)); // 날짜 형식 맞추기

        SupportProgram savedSupportProgram = supportProgramService.saveSupportProgram(supportProgram);
        return ResponseEntity.status(HttpStatus.CREATED).body(savedSupportProgram);
    }

    @GetMapping("/{id}")
    public ResponseEntity<SupportProgram> getSupportProgramById(@PathVariable Long id) {
        return supportProgramService.findSupportProgramById(id)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @GetMapping
    public ResponseEntity<List<SupportProgram>> getAllSupportPrograms() {
        List<SupportProgram> supportPrograms = supportProgramService.findAllSupportPrograms();
        return supportPrograms.isEmpty() ? ResponseEntity.notFound().build() : ResponseEntity.ok(supportPrograms);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteSupportProgram(@PathVariable Long id) {
        if (supportProgramService.findSupportProgramById(id).isPresent()) {
            supportProgramService.deleteSupportProgramById(id);
            return ResponseEntity.noContent().build(); // 204 No Content
        } else {
            return ResponseEntity.notFound().build(); // 404 Not Found
        }
    }
}
