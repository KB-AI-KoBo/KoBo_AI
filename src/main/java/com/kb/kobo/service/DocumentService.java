package com.kb.kobo.service;

import com.kb.kobo.entity.Document;
import com.kb.kobo.entity.User;
import com.kb.kobo.repository.DocumentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
public class DocumentService {

    @Value("${upload.dir}")
    private String uploadDir;

    private final DocumentRepository documentRepository;
    private final UserService userService;

    @Autowired
    public DocumentService(DocumentRepository documentRepository, UserService userService) {
        this.documentRepository = documentRepository;
        this.userService = userService;
    }

    @Transactional
    public Document uploadAndSaveDocument(MultipartFile file, String username) throws IOException {
        Path uploadPath = Paths.get(uploadDir);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }

        String originalFilename = file.getOriginalFilename();
        if (originalFilename == null || !originalFilename.toLowerCase().endsWith(".pdf")) {
            throw new RuntimeException("파일 형식이 올바르지 않습니다. PDF 파일만 업로드할 수 있습니다.");
        }

        String uniqueFileName = UUID.randomUUID() + ".pdf";
        Path filePath = uploadPath.resolve(uniqueFileName);

        try (var inputStream = file.getInputStream()) {
            Files.copy(inputStream, filePath);
        }

        User currentUser = userService.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("사용자를 찾을 수 없습니다."));

        Document document = new Document();
        document.setUser(currentUser);
        document.setFileName(originalFilename);
        document.setFileType(Document.FileType.PDF);
        document.setFilePath(uniqueFileName);

        return documentRepository.save(document);
    }


    public Document saveDocument(Document document) {
        return documentRepository.save(document);
    }

    public Optional<Document> findDocumentById(Long id) {
        return documentRepository.findById(id);
    }

    public List<Document> findDocumentsByUser(User user) {
        return documentRepository.findByUser(user);
    }

    @Transactional
    public void deleteDocument(Long documentId) {
        Document document = documentRepository.findById(documentId)
                .orElseThrow(() -> new RuntimeException("문서를 찾을 수 없습니다."));

        if (uploadDir == null || uploadDir.isEmpty()) {
            throw new RuntimeException("업로드 디렉토리가 설정되지 않았습니다.");
        }

        Path filePath = Paths.get(uploadDir).resolve(document.getFilePath());
        try {
            Files.deleteIfExists(filePath);
        } catch (IOException e) {
            throw new RuntimeException("파일 삭제 실패: " + e.getMessage());
        }

        documentRepository.deleteById(documentId);
    }
}
