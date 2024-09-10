package com.kb.kobo.validation;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;
import jakarta.validation.groups.Default;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;


@Target({ElementType.TYPE, ElementType.FIELD, ElementType.ANNOTATION_TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = {SignUpValidator.class})
public @interface ValidSignUp {

    String message() default "유효성 검사 실패";

    Class<?>[] groups() default {Default.class, ValidationGroups.SignUpValidation.class};

    Class<? extends Payload>[] payload() default {};
}
