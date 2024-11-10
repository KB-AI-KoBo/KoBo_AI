import React, { useState, useEffect } from 'react';
import '../styles/showP.css';
import '../styles/bootstrap.min.css';
import '../styles/style.css';

const ShowPrograms = () => {
    const [programs, setPrograms] = useState([]);
    const [currentPagePrograms, setCurrentPagePrograms] = useState([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [perPage, setPerPage] = useState(12); // 4x3 레이아웃을 위한 12개 표시
    const [pageGroup, setPageGroup] = useState(0); // 10단위 페이지 그룹 관리
    const apiKey = 'API 내놔!';

    useEffect(() => {
        const fetchPrograms = async () => {
            try {
                const response = await fetch(
                    `API 내놔!`,
                    {
                        headers: {
                            'Accept': '*/*'
                        }
                    }
                );

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                setPrograms(data.data);
                setTotalPages(Math.ceil(data.data.length / perPage)); // 전체 페이지 수 계산
                setCurrentPagePrograms(data.data.slice(0, perPage)); // 첫 페이지의 데이터 설정
            } catch (error) {
                console.error('Failed to fetch programs:', error);
            }
        };

        fetchPrograms();
    }, []);

    useEffect(() => {
        // 페이지가 변경될 때마다 현재 페이지의 데이터를 설정
        const startIndex = (page - 1) * perPage;
        const endIndex = startIndex + perPage;
        setCurrentPagePrograms(programs.slice(startIndex, endIndex));
        window.scrollTo(0, 0); // 페이지 변경 시 스크롤을 맨 위로 이동
    }, [page, programs, perPage]);

    const handleNextPageGroup = () => {
        if ((pageGroup + 1) * 10 < totalPages) {
            setPageGroup(pageGroup + 1);
            setPage(pageGroup * 10 + 11); // 다음 그룹의 첫 번째 페이지로 이동
        }
    };

    const handlePrevPageGroup = () => {
        if (pageGroup > 0) {
            setPageGroup(pageGroup - 1);
            setPage(pageGroup * 10 - 9); // 이전 그룹의 첫 번째 페이지로 이동
        }
    };

    return (
        <div className="program-list-container">
            <div className="program-list">
                {currentPagePrograms.map((program, index) => (
                    <div key={index} className="program-card">
                        <h3>{program.사업명}</h3>
                        <p><strong>분야:</strong> {program.분야}</p>
                        <p><strong>신청 기간:</strong> {program.신청시작일자} ~ {program.신청종료일자}</p>
                        <p><strong>소관 기관:</strong> {program.소관기관}</p>
                        <p><strong>수행 기관:</strong> {program.수행기관}</p>
                        <a href={program.상세URL} target="_blank" rel="noopener noreferrer">상세보기</a>
                    </div>
                ))}
            </div>
            <div className="pagination">
                <button onClick={handlePrevPageGroup} disabled={pageGroup === 0}>{'<<'}</button>
                {[...Array(10)].map((_, index) => {
                    const pageIndex = pageGroup * 10 + index + 1;
                    if (pageIndex <= totalPages) {
                        return (
                            <button
                                key={pageIndex}
                                className={page === pageIndex ? 'active' : ''}
                                onClick={() => setPage(pageIndex)}
                            >
                                {pageIndex}
                            </button>
                        );
                    }
                    return null;
                })}
                <button onClick={handleNextPageGroup} disabled={(pageGroup + 1) * 10 >= totalPages}>{'>>'}</button>
            </div>
        </div>
    );
};

export default ShowPrograms;