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
    const [분야, set분야] = useState('');  // 분야 필터링을 위한 상태 추가
    const [신청기간, set신청기간] = useState(''); // 신청 기간 필터링을 위한 상태 추가

    useEffect(() => {
        const fetchPrograms = async () => {
            try {
                const response = await fetch(
                    `https://api.odcloud.kr/api/3034791/v1/uddi:80a74cfd-55d2-4dd3-81c7-d01567d0b3c4?page=1&perPage=1000&serviceKey=dq0FiphIXegKyP%2F5zIDul95IvtalzdhixfdY7Hp9g4Onm%2FX9aCt378S5nVejoKY%2BGFEL5uvq75P1%2FlYuu%2Bf%2BLQ%3D%3D`,
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

        // 현재 날짜를 기준으로 신청 기간 상태 구분
        const today = new Date();

        const filteredPrograms = programs.filter((program) => {
            const startDate = new Date(program.신청시작일자);
            const endDate = new Date(program.신청종료일자);

            let 모집상태 = '';
            if (endDate < today) {
                모집상태 = '모집 완료';
            } else if (startDate <= today && today <= endDate) {
                모집상태 = '모집 중';
            } else if (startDate > today) {
                모집상태 = '모집 전';
            }

            return (
                (분야 === '' || program.분야 === 분야) &&
                (신청기간 === '' || 모집상태 === 신청기간)
            );
        });

        setCurrentPagePrograms(filteredPrograms.slice(startIndex, endIndex));
        window.scrollTo(0, 0); // 페이지 변경 시 스크롤을 맨 위로 이동
    }, [page, programs, perPage, 분야, 신청기간]);

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
            <div className="filter-container">
                <select value={분야} onChange={(e) => set분야(e.target.value)}>
                    <option value="">분야 선택</option>
                    <option value="경영">경영</option>
                    <option value="수출">수출</option>
                    <option value="인력">인력</option>
                    <option value="기술">기술</option>
                    <option value="내수">내수</option>
                    <option value="창업">창업</option>
                </select>

                <select value={신청기간} onChange={(e) => set신청기간(e.target.value)}>
                    <option value="">신청 기간 선택</option>
                    <option value="모집 전">모집 전</option>
                    <option value="모집 중">모집 중</option>
                    <option value="모집 완료">모집 완료</option>
                </select>
            </div>

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