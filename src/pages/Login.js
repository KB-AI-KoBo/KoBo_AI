import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/bootstrap.min.css';
import '../styles/style.css';
import '../styles/signup.css'; // 필요시 다른 스타일시트도 가져올 수 있음

const Login = () => {
    const navigate = useNavigate();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();

        const userData = {
            username: username,
            password: password,
        };

        try {
            const response = await fetch('http://localhost:5050/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userData),
            });

            if (response.ok) {

                const data = await response.json();
                if (data.token) {
                    localStorage.setItem('jwtToken', `Bearer ${data.token}`); // Bearer 붙여서 저장
                    navigate("/"); // 로그인 성공 후 메인 페이지로 리다이렉트
                } else {
                    alert("로그인 실패: JWT 토큰이 반환되지 않았습니다.");
                }
            } else {
                alert("로그인 실패: " + response.statusText);
            }
        } catch (error) {
            console.error('Error:', error);
            alert("로그인 중 오류가 발생했습니다.");
        }
    };

    return (
        <div className="container-fluid position-relative bg-white d-flex p-0">
            <div className="container-fluid">
                <div className="row h-100 align-items-center justify-content-center" style={{ minHeight: '100vh' }}>
                    <div className="col-12 col-sm-8 col-md-6 col-lg-5 col-xl-4">
                        <div className="bg-light rounded p-4 p-sm-5 my-4 mx-3">
                            <div className="d-flex align-items-center justify-content-between mb-3">
                                <h2 className="text-primary">KoBo</h2>
                                <h4>로그인</h4>
                            </div>
                            <form onSubmit={handleLogin}>
                                <div className="form-floating mb-3">
                                    <input
                                        type="text"
                                        className="form-control"
                                        id="floatingInput"
                                        placeholder="사용자 이름"
                                        value={username}
                                        onChange={(e) => setUsername(e.target.value)}
                                        required
                                    />
                                    <label htmlFor="floatingInput">사용자 이름</label>
                                </div>
                                <div className="form-floating mb-4">
                                    <input
                                        type="password"
                                        className="form-control"
                                        id="floatingPassword"
                                        placeholder="Password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        required
                                    />
                                    <label htmlFor="floatingPassword">비밀번호</label>
                                </div>
                                <div className="d-flex align-items-center justify-content-between mb-4">
                                    <div className="form-check">
                                        <input type="checkbox" className="form-check-input" id="exampleCheck1"/>
                                        <label className="form-check-label" htmlFor="exampleCheck1">아이디 기억하기</label>
                                    </div>
                                    <a href="/">비밀번호 찾기</a>
                                </div>
                                <button type="submit" className="btn btn-primary py-3 w-100 mb-4">로그인</button>
                            </form>
                            <p className="text-center mb-0">KoBo가 처음이신가요? <a href="/signup">회원가입</a></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;
