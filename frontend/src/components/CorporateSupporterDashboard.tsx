import React from 'react';
import { useUserType } from '../contexts/UserTypeContext';
import './Dashboard.css';

const CorporateSupporterDashboard: React.FC = () => {
  const { userType } = useUserType();

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>企業支援者ダッシュボード</h1>
        <p>Welcome, Corporate Supporter! 🏢</p>
        <p>現在のユーザータイプ: {userType}</p>
      </div>
      
      <div className="dashboard-content">
        <div className="dashboard-card">
          <h3>企業支援プログラム</h3>
          <p>企業として学生を支援しましょう</p>
          <button className="dashboard-button">プログラム作成</button>
        </div>
        
        <div className="dashboard-card">
          <h3>支援統計</h3>
          <p>企業の支援活動の統計を確認できます</p>
          <button className="dashboard-button">統計を見る</button>
        </div>
        
        <div className="dashboard-card">
          <h3>企業プロフィール</h3>
          <p>企業プロフィールを管理できます</p>
          <button className="dashboard-button">プロフィール</button>
        </div>
      </div>
    </div>
  );
};

export default CorporateSupporterDashboard;