import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import MainContent from './components/MainContent';
import Chat from './pages/Chat';
import SuggestedPrograms from './pages/SuggestedPrograms';
import ShowPrograms from './pages/ShowPrograms';
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Signup from './pages/Signup';

function App() {
    return (
        <Router>
            <Routes>
                {/* 로그인과 회원가입 경로는 사이드바와 메인 콘텐츠 없이 표시 */}
                <Route path="/login" element={<Login />} />
                <Route path="/signup" element={<Signup />} />

                {/* 사이드바와 메인 콘텐츠가 필요한 경로들 */}
                <Route
                    path="/*"
                    element={
                        <div className="d-flex">
                            <Sidebar />
                            <MainContent>
                                <Routes>
                                    <Route path="/" element={<Navigate to="/chat" />} /> {/* 기본적으로 재무분석 선택 */}
                                    <Route path="/chat" element={<Chat />} />
                                    <Route path="/suggested-programs" element={<SuggestedPrograms />} />
                                    <Route path="/show-programs" element={<ShowPrograms />} />
                                    <Route path="/dashboard" element={<Dashboard />} />
                                    <Route path="/login" element={<Login />} />
                                    <Route path="/signup" element={<Signup />} />
                                </Routes>
                            </MainContent>
                        </div>
                    }
                />
            </Routes>
        </Router>
    );
}

export default App;
