import React from 'react';
import ReactDOM from 'react-dom/client'; // 올바른 경로에서 createRoot를 임포트합니다.
import 'dotenv/config';
import App from './App';
import './styles/bootstrap.min.css';
import './styles/style.css';
import './styles/signup.css';
import './styles/showP.css';
import './styles/chat.css';


const root = ReactDOM.createRoot(document.getElementById('root')); // createRoot를 사용하여 React 18 호환성을 보장합니다.
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);
