import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/bootstrap.min.css';
import '../styles/style.css';
import '../styles/signup.css';

const Signup = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        email: '',
        password: '',
        username: '',
        companyName: '',
        industry: '',
        companySize: '',
        registrationNumber: '',
        companyEmail: ''
    });

    const handleChange = (e) => {
        const { id, value } = e.target;
        setFormData(prevState => ({
            ...prevState,
            [id]: value
        }));
    };

    const handleSignup = async () => {
        try {
            const response = await fetch('http://localhost:5050/user/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            });

            if (response.ok) {
                alert('회원가입이 완료되었습니다!');
                navigate('/login'); // 회원가입 성공 후 로그인 페이지로 이동
            } else {
                alert('회원가입 실패: ' + response.statusText);
            }
        } catch (error) {
            console.error('에러 발생:', error);
            alert('회원가입에 실패했습니다. 다시 시도해주세요.');
        }
    };

    return (
        <div className="container-fluid position-relative bg-white d-flex p-0">
            <div className="container-fluid">
                <div className="row h-100 align-items-center justify-content-center" style={{ minHeight: '100vh' }}>
                    <div className="col-12 col-sm-10 col-md-8 col-lg-7 col-xl-6">
                        <div className="bg-light rounded px-4 py-2 px-sm-5 py-sm-4 my-4 mx-3">
                            <div className="d-flex align-items-center justify-content-between mb-3">
                                <h3 className="text-primary">KoBo</h3>
                                <h5>회원가입</h5>
                            </div>
                            <div className="row h-100">
                                <div className="col-md-6">
                                    <div className="form-section left-section">
                                        <h5 className="text-primary form-title">사용자 정보</h5>
                                        <div className="form-floating mb-3">
                                            <input
                                                type="email"
                                                className="form-control"
                                                id="email"
                                                placeholder="name@example.com"
                                                value={formData.email}
                                                onChange={handleChange}
                                                required
                                            />
                                            <label htmlFor="email">이메일 주소</label>
                                        </div>
                                        <div className="form-floating mb-3">
                                            <input
                                                type="password"
                                                className="form-control"
                                                id="password"
                                                placeholder="Password"
                                                value={formData.password}
                                                onChange={handleChange}
                                                required
                                            />
                                            <label htmlFor="password">비밀번호</label>
                                        </div>
                                        <div className="form-floating mb-3">
                                            <input
                                                type="text"
                                                className="form-control"
                                                id="username"
                                                placeholder="name"
                                                value={formData.username}
                                                onChange={handleChange}
                                                required
                                            />
                                            <label htmlFor="username">이름</label>
                                        </div>
                                    </div>
                                </div>
                                <div className="col-md-6">
                                    <div className="form-section right-section">
                                        <h5 className="text-primary form-title">회사 정보</h5>
                                        <div className="form-floating mb-3">
                                            <input
                                                type="text"
                                                className="form-control"
                                                id="companyName"
                                                placeholder="회사이름"
                                                value={formData.companyName}
                                                onChange={handleChange}
                                                required
                                            />
                                            <label htmlFor="companyName">회사 이름</label>
                                        </div>
                                        <div className="form-floating mb-3">
                                            <input
                                                type="text"
                                                className="form-control"
                                                id="industry"
                                                placeholder="산업 분야"
                                                value={formData.industry}
                                                onChange={handleChange}
                                                required
                                            />
                                            <label htmlFor="industry">산업분야</label>
                                        </div>
                                        <div className="form-floating mb-3">
                                            <select
                                                className="form-select"
                                                id="companySize"
                                                aria-label="회사 구분"
                                                value={formData.companySize}
                                                onChange={handleChange}
                                                required
                                            >
                                                <option value="" disabled>회사 구분을 선택하세요</option>
                                                <option value="대기업">대기업</option>
                                                <option value="중견기업">중견기업</option>
                                                <option value="중소기업">중소기업</option>
                                                <option value="공기업">공기업</option>
                                                <option value="벤처기업">벤처기업</option>
                                            </select>
                                            <label htmlFor="companySize">회사 구분</label>
                                        </div>
                                        <div className="form-floating mb-3">
                                            <input
                                                type="text"
                                                className="form-control"
                                                id="registrationNumber"
                                                placeholder="사업자 번호"
                                                value={formData.registrationNumber}
                                                onChange={handleChange}
                                                required
                                            />
                                            <label htmlFor="registrationNumber">사업자 번호</label>
                                        </div>
                                        <div className="form-floating mb-3">
                                            <input
                                                type="email"
                                                className="form-control"
                                                id="companyEmail"
                                                placeholder="name@example.com"
                                                value={formData.companyEmail}
                                                onChange={handleChange}
                                                required
                                            />
                                            <label htmlFor="companyEmail">회사 이메일</label>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <button
                                type="button"
                                className="btn btn-primary py-3 w-100 custom-btn-color"
                                onClick={handleSignup}
                            >
                                회원가입
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Signup;
