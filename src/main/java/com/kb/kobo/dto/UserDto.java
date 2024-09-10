package com.kb.kobo.dto;

import com.kb.kobo.entity.User;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class UserDto {

    private Long id;

    @NotBlank(message = "사용자 이름은 필수입니다.")
    private String username;

    @NotBlank(message = "비밀번호는 필수입니다.")
    @Size(min = 6, message = "비밀번호는 최소 6글자 이상이어야 합니다.")
    private String password;

    @NotBlank(message = "이메일은 필수입니다.")
    @Email(message = "올바른 이메일 형식이어야 합니다.")
    private String email;

    @NotBlank(message = "회사 이름은 필수입니다.")
    private String companyName;

    @NotBlank(message = "회사 규모는 필수입니다.")
    private String companySize;

    @NotBlank(message = "사업자 등록 번호는 필수입니다.")
    private String registrationNumber;

    @NotBlank(message = "회사 이메일은 필수입니다.")
    @Email(message = "올바른 이메일 형식이어야 합니다.")
    private String companyEmail;

    @NotBlank(message = "산업 분야는 필수입니다.")
    private String industry;
}
