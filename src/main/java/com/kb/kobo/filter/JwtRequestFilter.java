package com.kb.kobo.filter;

import com.kb.kobo.authentication.UserEmailPasswordAuthenticationToken;
import com.kb.kobo.service.CustomUserDetailsService;
import com.kb.kobo.util.JwtUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Lazy;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;

import javax.servlet.FilterChain;
import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class JwtRequestFilter implements javax.servlet.Filter {

    private final CustomUserDetailsService customUserDetailsService;
    private final JwtUtil jwtUtil;

    @Autowired
    public JwtRequestFilter(CustomUserDetailsService customUserDetailsService, @Lazy JwtUtil jwtUtil) {
        this.customUserDetailsService = customUserDetailsService;
        this.jwtUtil = jwtUtil;
    }

    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        // 필터 초기화 코드 (필요시 구현)
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) request;
        HttpServletResponse httpResponse = (HttpServletResponse) response;

        final String authorizationHeader = httpRequest.getHeader("Authorization");

        String username = null;
        String jwtToken = null;

        if (authorizationHeader != null && authorizationHeader.startsWith("Bearer ")) {
            jwtToken = authorizationHeader.substring(7); // "Bearer " 제거
            username = jwtUtil.extractUsername(jwtToken); // 사용자 이름 추출
        }

        if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {
            UserDetails userDetails = customUserDetailsService.loadUserByUsername(username); // 사용자 이름으로 로드

            if (jwtUtil.validateToken(jwtToken, username)) {
                UserEmailPasswordAuthenticationToken authenticationToken =
                        new UserEmailPasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities());

                SecurityContextHolder.getContext().setAuthentication(authenticationToken);
            }
        }

        chain.doFilter(request, response); // 다음 필터 체인으로 요청 전달
    }

    @Override
    public void destroy() {
        // 필터 종료 코드 (필요시 구현)
    }
}
