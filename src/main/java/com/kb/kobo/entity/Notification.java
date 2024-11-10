package com.kb.kobo.entity;

import lombok.Getter;
import lombok.Setter;

import javax.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "Notification")
@Setter
@Getter
public class Notification {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_id")
    private Long userId;

    @Column(name = "is_Read", nullable = false)
    private boolean isRead;

    @Column(name = "message")
    private String message;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }

    // 기본 생성자
    public Notification() {
        // 기본 생성자는 JPA 요구 사항에 따라 필요합니다.
    }

    // 메시지를 받아 초기화하는 생성자
    public Notification(String message) {
        this.message = message;
        this.isRead = false; // 기본값은 읽지 않음
        this.createdAt = LocalDateTime.now(); // LocalDateTime으로 초기화
    }

    // 전체 필드를 받아 초기화하는 생성자
    public Notification(Long userId, String message, boolean isRead, LocalDateTime createdAt) {
        this.userId = userId;
        this.message = message;
        this.isRead = isRead;
        this.createdAt = createdAt; // LocalDateTime으로 초기화
    }
}
