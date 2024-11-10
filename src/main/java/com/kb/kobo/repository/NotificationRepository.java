package com.kb.kobo.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.kb.kobo.entity.Notification;


@Repository
public interface NotificationRepository extends JpaRepository<Notification, Long> {
}