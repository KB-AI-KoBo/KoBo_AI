import React, { useState, useRef, useEffect } from 'react';
import '../styles/chat.css';
import '../styles/bootstrap.min.css';
import '../styles/style.css';

// Base64 URL 디코딩 함수
const base64UrlDecode = (base64Url) => {
    let base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    return atob(base64);
};

// JWT 응답 디코딩 함수
const decodeJwtResponse = (token) => {
    let base64Payload = token.split('.')[1];
    let payload = base64UrlDecode(base64Payload);
    console.log('디코딩된 페이로드:', payload);
    return JSON.parse(payload);
};

// 토큰에서 이메일 추출하는 함수
const extractEmailFromToken = (token) => {
    try {
        const decodedToken = decodeJwtResponse(token.replace('Bearer ', '')); // 'Bearer ' 제거
        return decodedToken.email || decodedToken.sub;
    } catch (error) {
        console.error('JWT 토큰 디코딩 중 오류 발생:', error);
        return null;
    }
};

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [messageText, setMessageText] = useState('');
    const [attachedFile, setAttachedFile] = useState(null);
    const messagesContainerRef = useRef(null);

    // 텍스트 입력창 크기 자동 조절
    const handleInputChange = (event) => {
        setMessageText(event.target.value);
        event.target.style.height = 'auto';
        event.target.style.height = `${event.target.scrollHeight}px`;
    };

    // 파일 선택 처리
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setAttachedFile(file);
            setMessageText((prev) => prev + `[파일: ${file.name}] `);
        }
    };

    // 메시지 컨테이너의 마지막 메시지로 스크롤
    const scrollToBottom = () => {
        if (messagesContainerRef.current) {
            messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
        }
    };

    // 토큰이 만료되었는지 확인하고 필요시 새로 발급받기
    const checkAndRefreshToken = async () => {
        const authToken = localStorage.getItem('jwtToken');
        const refreshToken = localStorage.getItem('refreshToken');

        if (authToken) {
            try {
                const decodedToken = decodeJwtResponse(authToken);
                const expirationTime = decodedToken.exp * 1000;
                const currentTime = Date.now();

                console.log('checkAndRefreshToken에서 디코딩 된 코드를 받아옴');

                if (currentTime >= expirationTime && refreshToken) {
                    const response = await fetch('http://localhost:5080/auth/refresh-token', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: new URLSearchParams({
                            refreshToken: refreshToken,
                        }),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        localStorage.setItem('jwtToken', 'Bearer ${data.jwtToken}');
                        localStorage.setItem('refreshToken', data.refreshToken);
                        return `Bearer ${data.jwtToken}`; // 'Bearer ' 포함
                    } else {
                        console.error('리프레시 토큰 요청 실패');
                        localStorage.removeItem('jwtToken');
                        localStorage.removeItem('refreshToken');
                    }
                }
            } catch (e) {
                console.error('토큰 디코딩 중 오류 발생:', e);
            }
        }
        return authToken; // 'Bearer '가 포함된 authToken 반환
    };

    // 메시지 전송 처리
    const handleSendMessage = async () => {
        if (messageText.trim() || attachedFile) {
            // 사용자 메시지 상태 업데이트
            const userMessage = {
                type: 'user',
                content: messageText,
                file: attachedFile,
            };
            setMessages((prev) => [...prev, userMessage]);
            setMessageText('');
            setAttachedFile(null);
            scrollToBottom();

            // "분석중..." 메시지 표시
            const botMessage = {
                type: 'bot',
                content: '분석중...',
            };
            setMessages((prev) => [...prev, botMessage]);
            scrollToBottom();

            let analysisResponse = null;
            let questionId = null;

            // 유효한 토큰 확인 및 재발급
            const validToken = await checkAndRefreshToken();

            try {
                let uploadResult = null;

                if (attachedFile) {
                    // 파일이 첨부된 경우, 파일을 먼저 업로드
                    const uploadResult = await uploadFile(attachedFile, validToken);

                    try {
                        // 파일 업로드 성공 시, 질문과 문서 ID를 포함하여 제출
                        questionId = await submitQuestion(messageText, validToken, uploadResult.documentId);
                    } catch (error) {
                        throw new Error('질문 제출에 실패했습니다.');
                    }
                } else {
                    // 파일이 없는 경우, 질문만 제출
                    questionId = await submitQuestion(messageText, validToken);
                }

                if (questionId) {
                    // 질문 ID와 문서 ID로 분석 요청
                    analysisResponse = await submitAnalysisRequest(questionId, attachedFile ? uploadResult?.documentId : null);
                }

                // "분석중..." 메시지 제거
                setMessages((prev) => prev.slice(0, -1));
                if (analysisResponse) {
                    const botResponseMessage = {
                        type: 'bot',
                        content: analysisResponse.result,
                    };
                    setMessages((prev) => [...prev, botResponseMessage]);
                    scrollToBottom();
                }
            } catch (error) {
                console.error('오류 발생:', error);
                // 오류 메시지 표시
                const errorMessage = {
                    type: 'bot',
                    content: '메시지 전송 중 오류가 발생했습니다. 다시 시도해 주세요.',
                };
                setMessages((prev) => [...prev, errorMessage]);
                scrollToBottom();
            }
        }
    };


    // Enter 키로 메시지 전송
    useEffect(() => {
        const handleEnterPress = (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleSendMessage();
            }
        };

        document.addEventListener('keypress', handleEnterPress);
        return () => {
            document.removeEventListener('keypress', handleEnterPress);
        };
    }, [messageText, attachedFile]);


    // 파일 업로드 함수
    const uploadFile = async (file, authToken) => {
        if (!authToken) {
            console.error('토큰이 없습니다.');
            return null;
        }

        const email = extractEmailFromToken(authToken);
        console.log('Extracted email from token:', email);

        if (!email) {
            console.error('Email could not be extracted from token');
            return null;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('email', email);

        try {
            const response = await fetch('http://localhost:5050/api/documents/upload', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${authToken}`,
                },
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to upload file: ${response.status} ${response.statusText}. ${errorText}`);
            }

            // Document 객체의 ID와 파일 경로를 포함하는 응답을 반환
            const data = await response.json();
            console.log('File uploaded successfully:', data);

            if (data && data.documentId) {
                return {
                    documentId: data.documentId,
                    filePath: data.filePath
                };
            } else {
                console.error('Response data does not contain documentId');
                return null;
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            return null;
        }
    };

    // 질문 제출 함수
    const submitQuestion = async (messageText, authToken, documentId = null) => {
        if (!authToken) {
            console.error('토큰이 없습니다.');
            return null;
        }

        try {
            const requestBody = new URLSearchParams({
                documentId: documentId || '',
                content: messageText
            }).toString();

            const response = await fetch('http://localhost:5050/api/questions/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Authorization': `Bearer ${authToken}`,
                },
                body: requestBody,
            });

            if (response.ok) {
                const data = await response.json();
                console.log('Question submitted successfully:', data);
                return data.questionId;
            } else {
                const errorText = await response.text();
                console.error(`Failed to submit question: ${response.status} ${response.statusText}. ${errorText}`);
            }
        } catch (error) {
            console.error('Error submitting question:', error);
        }
        return null;
    };

    // // 분석 요청 함수
    // const submitAnalysisRequest = async (questionContent, file) => {
    //     try {
    //         const formData = new FormData();
    //         formData.append('query', questionContent);
    //         if (file) {
    //             formData.append('file', file);
    //         }
    //
    //         console.log('formData : ', questionContent, file);
    //
    //         const response = await fetch('http://localhost:5050/analysis/analyze', {
    //             method: 'POST',
    //             body: formData,
    //         });
    //
    //         if (response.ok) {
    //             const data = await response.json();
    //             console.log('분석 결과를 성공적으로 받았습니다:', data);
    //             return data;
    //         } else {
    //             console.error('분석 결과를 받는 데 실패했습니다:', response.statusText);
    //         }
    //     } catch (error) {
    //         console.error('분석 결과를 받는 도중 오류가 발생했습니다:', error);
    //     }
    //     return null;
    // };

    // 분석 결과 요청 함수
    const submitAnalysisRequest = async (questionContent, documentId) => {
        try {
            const requestBody = {
                documentId: documentId,
                content: questionContent
            };

            console.log('요청 데이터:', requestBody);

            const response = await fetch('http://localhost:5050/api/questions/submit', { // 백엔드의 submitQuestion 엔드포인트로 요청
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (response.ok) {
                const data = await response.json();
                console.log('분석 결과를 성공적으로 받았습니다:', data);
                return data;
            } else {
                console.error('분석 결과를 받는 데 실패했습니다:', response.statusText);
            }
        } catch (error) {
            console.error('분석 결과를 받는 도중 오류가 발생했습니다:', error);
        }
        return null;
    };


    return (
        <div className="container">
            <div className="messages-container" ref={messagesContainerRef}>
                {messages.map((message, index) => (
                    <div key={index} className={`message ${message.type}-message`}>
                        {message.content}
                        {message.file && (
                            <>
                                <br />
                                <a href={URL.createObjectURL(message.file)} download={message.file.name} className="file-link">
                                    {message.file.name}
                                </a>
                            </>
                        )}
                    </div>
                ))}
            </div>
            <div className="chat-container">
                <label htmlFor="file-input" className="attach-icon"><i className="fi fi-rs-folder"></i></label>
                <input type="file" id="file-input" style={{ display: 'none' }} onChange={handleFileChange} />
                <textarea
                    className="chat-input"
                    name="query"
                    rows="1"
                    placeholder="질문입력"
                    value={messageText}
                    onChange={handleInputChange}
                />
                <button className="send-button" onClick={handleSendMessage}>
                    <i className="fi fi-rs-paper-plane"></i>
                </button>
            </div>
        </div>
    );
};

export default Chat;
