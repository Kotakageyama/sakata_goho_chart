import React from 'react';
import { useUserType } from '../contexts/UserTypeContext';
import './Dashboard.css';

const IndividualSupporterDashboard: React.FC = () => {
  const { userType } = useUserType();

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>個人支援者ダッシュボード</h1>
        <p>Welcome, Individual Supporter! 👤</p>
        <p>現在のユーザータイプ: {userType}</p>
      </div>
      
      <div className="dashboard-content">
        <div className="dashboard-card">
          <h3>学生を支援</h3>
          <p>学生に奨学金を提供しましょう</p>
          <button className="dashboard-button">支援する</button>
        </div>
        
        <div className="dashboard-card">
          <h3>支援履歴</h3>
          <p>これまでの支援活動を確認できます</p>
          <button className="dashboard-button">履歴を見る</button>
        </div>
        
        <div className="dashboard-card">
          <h3>支援者プロフィール</h3>
          <p>あなたの支援者プロフィールを管理できます</p>
          <button className="dashboard-button">プロフィール</button>
        </div>
      </div>
    </div>
  );
};

export default IndividualSupporterDashboard;