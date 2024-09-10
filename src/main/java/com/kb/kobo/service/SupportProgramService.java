package com.kb.kobo.service;

import com.kb.kobo.entity.SupportProgram;
import com.kb.kobo.repository.SupportProgramRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class SupportProgramService {

    private final SupportProgramRepository supportProgramRepository;

    @Autowired
    public SupportProgramService(SupportProgramRepository supportProgramRepository) {
        this.supportProgramRepository = supportProgramRepository;
    }

    public SupportProgram saveSupportProgram(SupportProgram supportProgram) {
        return supportProgramRepository.save(supportProgram);
    }

    public Optional<SupportProgram> findSupportProgramById(Long id) {
        return supportProgramRepository.findById(id);
    }

    public List<SupportProgram> findAllSupportPrograms() {
        return supportProgramRepository.findAll();
    }

    public void deleteSupportProgramById(Long id) {
        supportProgramRepository.deleteById(id);
    }
}

