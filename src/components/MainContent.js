import React, { useEffect, useState } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Chat from '../pages/Chat';
import ShowPrograms from '../pages/ShowPrograms';  // ShowPrograms 컴포넌트 임포트
import '../styles/style.css';
import '../styles/bootstrap.min.css';
import '../styles/chat.css';

const MainContent = () => {
    const [loading, setLoading] = useState(true);
    const location = useLocation();

    useEffect(() => {
        const spinnerTimeout = setTimeout(() => {
            setLoading(false);
        }, 1000); // 1초 후 스피너 숨김

        const handleScroll = () => {
            if (window.scrollY > 300) {
                document.querySelector('.back-to-top').style.display = 'block';
            } else {
                document.querySelector('.back-to-top').style.display = 'none';
            }
        };

        window.addEventListener('scroll', handleScroll);

        return () => {
            clearTimeout(spinnerTimeout);
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);

    const scrollToTop = () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    };

    const toggleSidebar = () => {
        document.querySelector('.sidebar').classList.toggle('open');
        document.querySelector('.content').classList.toggle('open');
    };

    return (
        <div className="content">
            <nav className="navbar sticky-top px-1 py-3">
                <a href="#" className="sidebar-toggler flex-shrink-0" onClick={toggleSidebar}>
                    <i className="fa fa-bars"></i>
                </a>
            </nav>
            {loading && (
                <div id="spinner" className="show bg-white position-fixed translate-middle w-100 vh-100 top-50 start-50 d-flex align-items-center justify-content-center">
                    <div className="spinner-border text-primary" style={{ width: '3rem', height: '3rem' }} role="status">
                        <span className="sr-only">Loading...</span>
                    </div>
                </div>
            )}
            <div className="container-fluid pt-4 px-4" style={{ height: 'calc(100vh - 56px)', display: 'flex', flexDirection: 'column' }}>
                <div id="main-content" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                    <Routes>
                        <Route path="/chat" element={<Chat />} />  {/* 경로가 /chat일 때 Chat 컴포넌트를 렌더링 */}
                        <Route path="/show-programs" element={<ShowPrograms />} /> {/* 경로가 /show-programs일 때 ShowPrograms 컴포넌트를 렌더링 */}
                    </Routes>
                </div>
            </div>
            <button className="back-to-top btn btn-lg btn-primary btn-lg-square" onClick={scrollToTop} style={{ display: 'none' }}>
                <i className="bi bi-arrow-up"></i>
            </button>
        </div>
    );
};

export default MainContent;
