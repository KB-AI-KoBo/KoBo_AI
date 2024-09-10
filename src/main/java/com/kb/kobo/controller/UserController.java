package com.kb.kobo.controller;

import com.kb.kobo.dto.UserDto;
import com.kb.kobo.entity.User;
import com.kb.kobo.service.UserService;
import com.kb.kobo.util.JwtUtil;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.Optional;

@RestController
@CrossOrigin(origins = "http://localhost:3000")
@RequestMapping("/api/user")
public class UserController {

    private final UserService userService;
    private final JwtUtil jwtUtil;

    @Autowired
    public UserController(UserService userService, JwtUtil jwtUtil) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
    }

    @PostMapping("/signup")
    public ResponseEntity<User> signup(@Valid @RequestBody UserDto userDto) {
        User savedUser = userService.signUp(userDto);
        return ResponseEntity.status(HttpStatus.CREATED).body(savedUser);
    }

    @GetMapping("/profile")
    public ResponseEntity<User> getCurrentUser(HttpServletRequest request) {
        // JWT 토큰을 요청 헤더에서 가져오기
        String token = request.getHeader("Authorization");
        if (token != null && token.startsWith("Bearer ")) {
            token = token.substring(7); // "Bearer "를 제거
        }

        if (token == null || jwtUtil.isTokenExpired(token)) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }

        // JWT에서 username을 추출
        String username = jwtUtil.extractUsername(token);

        // 사용자 정보를 서비스에서 조회합니다.
        Optional<User> userOptional = userService.findByUsername(username);
        return userOptional.map(ResponseEntity::ok).orElseGet(() -> ResponseEntity.notFound().build());
    }

    @GetMapping("/profile/username/{username}")
    public ResponseEntity<User> findByUsername(@PathVariable String username) {
        Optional<User> userOptional = userService.findByUsername(username);
        return userOptional.map(ResponseEntity::ok).orElseGet(() -> ResponseEntity.notFound().build());
    }

    @GetMapping("/profile/email/{email}")
    public ResponseEntity<User> findByEmail(@PathVariable String email) {
        Optional<User> userOptional = userService.findByEmail(email);
        return userOptional.map(ResponseEntity::ok).orElseGet(() -> ResponseEntity.notFound().build());
    }

    @PostMapping("/profile")
    public ResponseEntity<User> updateUserProfile(@Valid @RequestBody UserDto updatedUserDto) {
        // updatedUserDto에서 사용자의 식별자(ID)를 가져와서 사용
        Long userId = updatedUserDto.getId(); // 예를 들어, getId() 메서드를 사용

        User updatedUser = userService.updateUserProfile(userId, updatedUserDto);
        return ResponseEntity.ok(updatedUser);
    }
}
