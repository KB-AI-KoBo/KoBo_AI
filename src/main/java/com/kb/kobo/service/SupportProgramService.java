package com.kb.kobo.service;

import com.kb.kobo.dto.ProgramResponse;
import com.kb.kobo.entity.Program;
import com.kb.kobo.repository.SupportProgramRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class SupportProgramService {

    private final String API_URL = "https://api.odcloud.kr/api/3034791/v1/uddi:80a74cfd-55d2-4dd3-81c7-d01567d0b3c4?serviceKey=dq0FiphIXegKyP%2F5zIDul95IvtalzdhixfdY7Hp9g4Onm%2FX9aCt378S5nVejoKY%2BGFEL5uvq75P1%2FlYuu%2Bf%2BLQ%3D%3D";

    public List<Program> getFilteredPrograms(String 분야, String 소관기관, String 신청시작일자, String 신청종료일자){
        RestTemplate restTemplate = new RestTemplate();
        ProgramResponse response = restTemplate.getForObject(API_URL, ProgramResponse.class);

        List<Program> programs = response.getData();

        // 필터링 로직
        if (분야 != null && !분야.isEmpty()) {
            programs = programs.stream()
                    .filter(program -> program.get분야().equals(분야))
                    .collect(Collectors.toList());
        }

        if (소관기관 != null && !소관기관.isEmpty()) {
            programs = programs.stream()
                    .filter(program -> program.get소관기관().equals(소관기관))
                    .collect(Collectors.toList());
        }

        LocalDate now = LocalDate.now();
        if (신청시작일자 != null && 신청종료일자 != null) {
            programs = programs.stream()
                    .filter(program -> {
                        LocalDate startDate = LocalDate.parse(program.get신청시작일자());
                        LocalDate endDate = LocalDate.parse(program.get신청종료일자());
                        return (startDate.isBefore(now) || startDate.isEqual(now)) && (endDate.isAfter(now) || endDate.isEqual(now)); // 진행 중인 것
                    })
                    .collect(Collectors.toList());
        }

        return programs;
    }
}
