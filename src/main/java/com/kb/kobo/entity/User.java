package com.kb.kobo.entity;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import javax.persistence.*;
import java.time.LocalDateTime;
import java.util.Collection;
import java.util.Collections;

@Entity
@Table(name = "users")
@Getter
@Setter
public class User implements UserDetails {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_id")
    private Long userId;

    @Column(name = "username", unique = true, nullable = false)
    @NotBlank(message = "사용자 이름은 필수입니다.")
    private String username;

    @Column(name = "password", nullable = false)
    @NotBlank(message = "비밀번호는 필수입니다.")
    @Size(min = 6, message = "비밀번호는 최소 6글자 이상이어야 합니다.")
    private String password;

    @Column(name = "email", unique = true, nullable = false)
    @NotBlank(message = "이메일은 필수입니다.")
    @Email(message = "올바른 이메일 형식이어야 합니다.")
    private String email;

    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    private LocalDateTime createdAt;

    @Column(name = "company_name", nullable = false)
    @NotBlank(message = "회사 이름은 필수입니다.")
    private String companyName;

    @Column(name = "company_size", nullable = false)
    private String companySize;

    @Column(name = "registration_number", unique = true, nullable = false)
    @NotBlank(message = "사업자 등록 번호는 필수입니다.")
    private String registrationNumber;

    @Column(name = "company_email", unique = true, nullable = false)
    @NotBlank(message = "회사 이메일은 필수입니다.")
    @Email(message = "올바른 이메일 형식이어야 합니다.")
    private String companyEmail;

    @Column(name = "industry", nullable = false)
    @NotBlank(message = "산업 분야는 필수입니다.")
    private String industry;


    public User() {
        this.createdAt = LocalDateTime.now(); // 현재 시각으로 설정
    }


    public User(String username, String password, String email, String companyName, String companySize, String registrationNumber, String companyEmail,String industry,  LocalDateTime createdAt) {
        this.username = username;
        this.password = password;
        this.email = email;
        this.companyName = companyName;
        this.companySize = companySize;
        this.registrationNumber = registrationNumber;
        this.companyEmail = companyEmail;
        this.industry = industry;
        this.createdAt = createdAt; // LocalDateTime 형식으로 설정
    }


    public User(String username, String password, String email) {
        this.username = username;
        this.password = password;
        this.email = email;
        this.createdAt = LocalDateTime.now(); // 현재 시각으로 설정
    }


//    // Enum for company size
//    public enum CompanySize {
//        MICRO, SMALL, MEDIUM, MIDSIZE, LARGE, VERY_LARGE
//    }

    @Override
    public String toString() {
        return "User{" +
                "userId=" + userId +
                ", username='" + username + '\'' +
                ", email='" + email + '\'' +
                ", createdAt=" + createdAt +
                ", companyName='" + companyName + '\'' +
                ", companySize=" + companySize +
                ", registrationNumber='" + registrationNumber + '\'' +
                ", companyEmail='" + companyEmail + '\'' +
                ", industry='" + industry + '\'' +
                '}';
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        // 사용자의 권한을 반환 (단일 권한 반환)
        return Collections.singletonList(new SimpleGrantedAuthority("USER"));
    }

    @Override
    public String getPassword() {
        return password;
    }

    @Override
    public String getUsername() {
        return username;
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }
}
