package com.kb.kobo.validation;

import com.kb.kobo.dto.UserDto;

import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;

public class SignUpValidator implements ConstraintValidator<ValidSignUp, UserDto> {

    @Override
    public void initialize(ValidSignUp constraintAnnotation) {
    }

    @Override
    public boolean isValid(UserDto userDto, ConstraintValidatorContext context) {
        if (userDto == null) {
            return false; // UserDto가 null인 경우 유효하지 않음
        }

        // 사용자 이름, 비밀번호, 이메일 필드가 null이 아닌지 검사
        if (userDto.getPassword() == null || userDto.getEmail() == null) {
            return false;
        }


        return true; // 모든 검사를 통과하면 유효함
    }
}
