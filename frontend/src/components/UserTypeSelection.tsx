import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useUserType, UserType } from '../contexts/UserTypeContext';
import './UserTypeSelection.css';

interface UserTypeCard {
  type: UserType;
  title: string;
  description: string;
  icon: string;
}

const UserTypeSelection: React.FC = () => {
  const navigate = useNavigate();
  const { setUserType } = useUserType();

  const userTypes: UserTypeCard[] = [
    {
      type: 'student',
      title: '奨学生',
      description: '世界中の学生として奨学金を受け取る',
      icon: '🎓'
    },
    {
      type: 'individual-supporter',
      title: '個人支援者',
      description: '個人として学生を支援する',
      icon: '👤'
    },
    {
      type: 'corporate-supporter',
      title: '企業支援者',
      description: '企業として学生を支援する',
      icon: '🏢'
    }
  ];

  const handleTypeSelection = (type: UserType) => {
    setUserType(type);
    navigate(`/${type}/dashboard`);
  };

  return (
    <div className="user-type-selection">
      <div className="header">
        <h1>ユーザータイプを選択してください</h1>
        <p>World ID認証が完了しました。どのようにシステムを利用しますか？</p>
      </div>
      
      <div className="cards-container">
        {userTypes.map((userType) => (
          <div
            key={userType.type}
            className="user-type-card"
            onClick={() => handleTypeSelection(userType.type)}
          >
            <div className="card-icon">{userType.icon}</div>
            <h3 className="card-title">{userType.title}</h3>
            <p className="card-description">{userType.description}</p>
            <button className="select-button">選択</button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default UserTypeSelection;