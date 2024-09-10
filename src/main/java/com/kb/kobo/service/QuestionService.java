package com.kb.kobo.service;

import com.kb.kobo.entity.Document;
import com.kb.kobo.entity.Question;
import com.kb.kobo.entity.User;
import com.kb.kobo.repository.QuestionRepository;
import com.kb.kobo.repository.DocumentRepository;
import com.kb.kobo.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

@Service
public class QuestionService {

    private final QuestionRepository questionRepository;
    private final DocumentRepository documentRepository;
    private final UserRepository userRepository;

    @Autowired
    public QuestionService(QuestionRepository questionRepository,
                           DocumentRepository documentRepository,
                           UserRepository userRepository) {
        this.questionRepository = questionRepository;
        this.documentRepository = documentRepository;
        this.userRepository = userRepository;
    }

    @Transactional
    public Question submitQuestion(Long documentId, String username, String content) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("사용자를 찾을 수 없습니다."));

        Question question = new Question();
        question.setUser(user);
        System.out.println("해당 사용자(user) : " + question.getUser().getUsername());
        question.setContent(content);
        System.out.println("저장할 질문 내용(content) : " + question.getContent());


        // documentId가 null이 아닐 경우에만 Document를 조회합니다.
        if (documentId != null) {
            Document document = documentRepository.findById(documentId)
                    .orElseThrow(() -> new RuntimeException("문서를 찾을 수 없습니다."));
            question.setDocumentId(document);
            System.out.println("저장된 문서(documentID) : " + question.getDocumentId());
        }

        return questionRepository.save(question);
    }


    public Optional<Question> findQuestionById(Long id) {
        return questionRepository.findById(id);
    }

    @Transactional
    public void deleteQuestion(Long id) {
        if (questionRepository.existsById(id)) {
            questionRepository.deleteById(id);
        } else {
            throw new RuntimeException("질문을 찾을 수 없습니다.");
        }
    }
}
