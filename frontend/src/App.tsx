import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import UserTypeSelection from './components/UserTypeSelection';
import StudentDashboard from './components/StudentDashboard';
import IndividualSupporterDashboard from './components/IndividualSupporterDashboard';
import CorporateSupporterDashboard from './components/CorporateSupporterDashboard';
import { UserTypeProvider } from './contexts/UserTypeContext';
import './App.css';

function App() {
  return (
    <UserTypeProvider>
      <div className="App">
        <Router>
          <Routes>
            <Route path="/" element={<Navigate to="/select-type" replace />} />
            <Route path="/select-type" element={<UserTypeSelection />} />
            <Route path="/student/dashboard" element={<StudentDashboard />} />
            <Route path="/individual-supporter/dashboard" element={<IndividualSupporterDashboard />} />
            <Route path="/corporate-supporter/dashboard" element={<CorporateSupporterDashboard />} />
          </Routes>
        </Router>
      </div>
    </UserTypeProvider>
  );
}

export default App;