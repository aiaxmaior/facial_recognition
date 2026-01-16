import React, { useState } from 'react';
import { EnrollmentModal, EnrollmentStatus, EnrollmentResult } from '@qryde/facial-enrollment';

// Mock employee data
interface Employee {
  id: string;
  name: string;
  email: string;
  employeeId: string;
  role: 'Admin' | 'User' | 'Supervisor' | 'Manager';
  avatar?: string;
  biometricStatus: EnrollmentStatus;
}

const mockEmployees: Employee[] = [
  { id: '1', name: 'Kelly Holmes', email: 'holmeskelly@mail.com', employeeId: 'E502', role: 'Admin', biometricStatus: 'enrolled' },
  { id: '2', name: 'Stephanie Sharkey', email: 'sharkeysteph@mail.com', employeeId: 'E502', role: 'Supervisor', biometricStatus: 'pending' },
  { id: '3', name: 'Dennis Callis', email: 'callisdennis@mail.com', employeeId: 'E502', role: 'Manager', biometricStatus: 'unenrolled' },
  { id: '4', name: 'Frances Swann', email: 'swanfrances@mail.com', employeeId: 'E502', role: 'User', biometricStatus: 'enrolled' },
  { id: '5', name: 'Daniel Hamilton', email: 'holmeskelly@mail.com', employeeId: 'E502', role: 'Manager', biometricStatus: 'unenrolled' },
  { id: '6', name: 'John Dukes', email: 'sharkeysteph@mail.com', employeeId: 'E502', role: 'Admin', biometricStatus: 'pending' },
  { id: '7', name: 'Katie Sims', email: 'callisdennis@mail.com', employeeId: 'E502', role: 'User', biometricStatus: 'enrolled' },
  { id: '8', name: 'James Hall', email: 'swanfrances@mail.com', employeeId: 'E502', role: 'Admin', biometricStatus: 'unenrolled' },
  { id: '9', name: 'Autumn Phillips', email: 'holmeskelly@mail.com', employeeId: 'E502', role: 'Supervisor', biometricStatus: 'enrolled' },
  { id: '10', name: 'Patricia Sanders', email: 'sharkeysteph@mail.com', employeeId: 'E502', role: 'Supervisor', biometricStatus: 'unenrolled' },
  { id: '11', name: 'Kimberly Mastrangelo', email: 'callisdennis@mail.com', employeeId: 'E502', role: 'Admin', biometricStatus: 'pending' },
  { id: '12', name: 'Jerry Helfer', email: 'swanfrances@mail.com', employeeId: 'E502', role: 'User', biometricStatus: 'enrolled' },
];

function App() {
  const [employees, setEmployees] = useState(mockEmployees);
  const [selectedEmployee, setSelectedEmployee] = useState<Employee | null>(null);
  const [isEnrollmentOpen, setIsEnrollmentOpen] = useState(false);

  const handleOpenEnrollment = (employee: Employee) => {
    setSelectedEmployee(employee);
    setIsEnrollmentOpen(true);
  };

  const handleCloseEnrollment = () => {
    setIsEnrollmentOpen(false);
    setSelectedEmployee(null);
  };

  const handleEnrollmentComplete = (result: EnrollmentResult) => {
    console.log('Enrollment complete:', result);
    
    // Update employee status
    if (result.success && selectedEmployee) {
      setEmployees((prev) =>
        prev.map((emp) =>
          emp.id === selectedEmployee.id
            ? { ...emp, biometricStatus: 'pending' as EnrollmentStatus }
            : emp
        )
      );
    }
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map((n) => n[0])
      .join('')
      .toUpperCase();
  };

  const getRoleBadgeClass = (role: string) => {
    const classes: Record<string, string> = {
      Admin: 'badge-admin',
      User: 'badge-user',
      Supervisor: 'badge-supervisor',
      Manager: 'badge-manager',
    };
    return classes[role] || 'badge-user';
  };

  const getBiometricBadgeClass = (status: EnrollmentStatus) => {
    const classes: Record<EnrollmentStatus, string> = {
      enrolled: 'biometric-enrolled',
      pending: 'biometric-pending',
      unenrolled: 'biometric-unenrolled',
    };
    return classes[status];
  };

  const getBiometricLabel = (status: EnrollmentStatus) => {
    const labels: Record<EnrollmentStatus, string> = {
      enrolled: '✓ Enrolled',
      pending: '⏳ Pending',
      unenrolled: '✗ Not Enrolled',
    };
    return labels[status];
  };

  const stats = {
    total: employees.length,
    active: employees.filter((e) => e.biometricStatus === 'enrolled').length,
    inactive: employees.filter((e) => e.biometricStatus !== 'enrolled').length,
    newThisMonth: 2,
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <svg viewBox="0 0 32 32" fill="currentColor">
            <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none" />
            <path d="M10 16l4 4 8-8" stroke="currentColor" strokeWidth="2" fill="none" />
          </svg>
          QRaie
        </div>
        
        <div className="search-bar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
          </svg>
          <input type="text" placeholder="Search employee" />
        </div>
        
        <div className="user-menu">
          <div className="user-avatar">MJ</div>
          <span>Mary Jones</span>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="m6 9 6 6 6-6" />
          </svg>
        </div>
      </header>

      <div className="main-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-item">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7" />
              <rect x="14" y="3" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" />
            </svg>
          </div>
          <div className="sidebar-item">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
          </div>
          <div className="sidebar-item active">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </div>
          <div className="sidebar-item">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </div>
        </aside>

        {/* Main Content */}
        <main className="content">
          {/* Tabs */}
          <div className="tabs">
            <button className="tab active">Employee</button>
            <button className="tab">Department</button>
            <button className="tab">Work Order</button>
          </div>

          {/* Stats */}
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                </svg>
              </div>
              <div className="stat-content">
                <div className="stat-label">Total <span className="highlight">Employee</span></div>
                <div className="stat-value">{stats.total}</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                  <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                </svg>
              </div>
              <div className="stat-content">
                <div className="stat-label">Active <span className="highlight">Employee</span></div>
                <div className="stat-value">{stats.active}</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                </svg>
              </div>
              <div className="stat-content">
                <div className="stat-label">InActive <span className="highlight">Employee</span></div>
                <div className="stat-value">{stats.inactive}</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2">
                  <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="8.5" cy="7" r="4" />
                  <line x1="20" y1="8" x2="20" y2="14" />
                  <line x1="23" y1="11" x2="17" y2="11" />
                </svg>
              </div>
              <div className="stat-content">
                <div className="stat-label">New <span className="highlight">Employee</span></div>
                <div className="stat-value">{stats.newThisMonth}</div>
              </div>
            </div>
          </div>

          {/* Employee List Header */}
          <div className="employee-header">
            <h2 className="employee-title">
              Employees Overview: <span className="count">{employees.length}</span>
            </h2>
            <div className="employee-actions">
              <button className="btn btn-outline">
                Department ▾
              </button>
              <button className="btn btn-outline">
                ↓ Import
              </button>
              <button className="btn btn-primary">
                + Add Employee
              </button>
            </div>
          </div>

          {/* Employee Grid */}
          <div className="employee-grid">
            {employees.map((employee) => (
              <div key={employee.id} className="employee-card">
                <div className="employee-card-header">
                  <div className="employee-avatar">
                    {employee.avatar ? (
                      <img src={employee.avatar} alt={employee.name} />
                    ) : (
                      getInitials(employee.name)
                    )}
                  </div>
                  <div className="employee-info">
                    <div className="employee-name">{employee.name}</div>
                    <div className="employee-email">{employee.email}</div>
                  </div>
                  <div className="employee-card-menu">⋮</div>
                </div>
                
                <div className="employee-meta">
                  <div className="employee-meta-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="2" y="5" width="20" height="14" rx="2" />
                      <path d="M2 10h20" />
                    </svg>
                    ID: {employee.employeeId}
                  </div>
                  <div className="employee-meta-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                      <circle cx="12" cy="7" r="4" />
                    </svg>
                    Role: <span className={`badge ${getRoleBadgeClass(employee.role)}`}>{employee.role}</span>
                  </div>
                </div>

                <div className="biometric-status">
                  <span className="biometric-label">Biometric:</span>
                  <span className={`biometric-badge ${getBiometricBadgeClass(employee.biometricStatus)}`}>
                    {getBiometricLabel(employee.biometricStatus)}
                  </span>
                  <button
                    className={`enroll-btn ${employee.biometricStatus === 'enrolled' ? 're-enroll' : ''}`}
                    onClick={() => handleOpenEnrollment(employee)}
                  >
                    {employee.biometricStatus === 'enrolled' ? 'Re-enroll' : 
                     employee.biometricStatus === 'pending' ? 'Complete' : 'Enroll'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </main>
      </div>

      {/* Enrollment Modal */}
      {selectedEmployee && (
        <EnrollmentModal
          isOpen={isEnrollmentOpen}
          onClose={handleCloseEnrollment}
          userId={selectedEmployee.id}
          userName={selectedEmployee.name}
          enrollmentStatus={selectedEmployee.biometricStatus}
          apiEndpoint="http://localhost:3001/api/enrollment"
          onEnrollmentComplete={handleEnrollmentComplete}
          enableAudio={true}
        />
      )}
    </div>
  );
}

export default App;
