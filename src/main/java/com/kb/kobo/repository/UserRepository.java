package com.kb.kobo.repository;

import com.kb.kobo.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    Optional<User> findByUsername(String username);

    Optional<User> findByEmail(String email);

    // 등록 번호 기준
    Optional<User> findByRegistrationNumber(String registrationNumber);

    Optional<User> findByCompanyEmail(String companyEmail);

    long countByIndustry(String industry);

    List<User> findByCompanySize(String companySize);

    List<User> findTop10ByOrderByCreatedAtDesc();

    void deleteByCompanyName(String companyName);


    List<User> findByCompanyEmailContaining(String companyEmail);
}
