import React, { createContext, useContext, useState, ReactNode } from 'react';

export type UserType = 'student' | 'individual-supporter' | 'corporate-supporter';

interface UserTypeContextType {
  userType: UserType | null;
  setUserType: (type: UserType) => void;
}

const UserTypeContext = createContext<UserTypeContextType | undefined>(undefined);

export const useUserType = () => {
  const context = useContext(UserTypeContext);
  if (context === undefined) {
    throw new Error('useUserType must be used within a UserTypeProvider');
  }
  return context;
};

interface UserTypeProviderProps {
  children: ReactNode;
}

export const UserTypeProvider: React.FC<UserTypeProviderProps> = ({ children }) => {
  const [userType, setUserType] = useState<UserType | null>(null);

  const value = {
    userType,
    setUserType,
  };

  return (
    <UserTypeContext.Provider value={value}>
      {children}
    </UserTypeContext.Provider>
  );
};