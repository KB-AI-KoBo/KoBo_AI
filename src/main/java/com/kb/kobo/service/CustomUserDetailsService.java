package com.kb.kobo.service;

import com.kb.kobo.entity.User;
import com.kb.kobo.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.ArrayList;

@Service
public class CustomUserDetailsService implements UserDetailsService {

    @Autowired
    private UserRepository userRepository; // 사용자 정보를 조회하는 레포지토리

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // username 대신 email로 변경하여 조회
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("해당 이름을 가진 유저가 존재하지 않습니다. 이름: " + username));
        return new org.springframework.security.core.userdetails.User(user.getUsername(), user.getPassword(), new ArrayList<>());
    }

}
