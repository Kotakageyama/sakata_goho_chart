import React from 'react';
import { useUserType } from '../contexts/UserTypeContext';
import './Dashboard.css';

const StudentDashboard: React.FC = () => {
  const { userType } = useUserType();

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>奨学生ダッシュボード</h1>
        <p>Welcome, Student! 📚</p>
        <p>現在のユーザータイプ: {userType}</p>
      </div>
      
      <div className="dashboard-content">
        <div className="dashboard-card">
          <h3>奨学金申請</h3>
          <p>新しい奨学金に申請しましょう</p>
          <button className="dashboard-button">申請する</button>
        </div>
        
        <div className="dashboard-card">
          <h3>申請状況</h3>
          <p>あなたの申請状況を確認できます</p>
          <button className="dashboard-button">確認する</button>
        </div>
        
        <div className="dashboard-card">
          <h3>受給履歴</h3>
          <p>過去の受給履歴を確認できます</p>
          <button className="dashboard-button">履歴を見る</button>
        </div>
      </div>
    </div>
  );
};

export default StudentDashboard;