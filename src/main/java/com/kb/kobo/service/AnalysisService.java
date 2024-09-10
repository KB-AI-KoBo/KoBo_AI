package com.kb.kobo.service;

import com.kb.kobo.entity.AnalysisResult;
import com.kb.kobo.repository.AnalysisResultRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Service
public class AnalysisService {

    private final AnalysisResultRepository analysisResultRepository;

    @Autowired
    public AnalysisService(AnalysisResultRepository analysisResultRepository) {
        this.analysisResultRepository = analysisResultRepository;
    }

    @Transactional
    public AnalysisResult saveAnalysisResult(AnalysisResult analysisResult) {
        return analysisResultRepository.save(analysisResult);
    }

    public Optional<AnalysisResult> findAnalysisResultById(Long id) {
        return analysisResultRepository.findById(id);
    }

    public List<AnalysisResult> findAnalysisResultsByDocumentId(Long documentId) {
        return analysisResultRepository.findByDocument_DocumentId(documentId);
    }

    public void deleteAnalysisResultById(Long id) {
        analysisResultRepository.deleteById(id);
    }
}
