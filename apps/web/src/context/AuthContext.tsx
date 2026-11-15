import React, { createContext, useContext, useState, useEffect } from 'react';
import type { User, LoginPayload, RegisterPayload } from '../types';
import { authApi } from '../api/client';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (payload: LoginPayload) => Promise<void>;
  register: (payload: RegisterPayload) => Promise<User>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('automl_token'));
  const [user, setUser] = useState<User | null>(() => {
    const saved = localStorage.getItem('automl_user');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        return null;
      }
    }
    return null;
  });
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const initAuth = async () => {
      const savedToken = localStorage.getItem('automl_token');
      if (savedToken) {
        try {
          const profile = await authApi.getMe();
          setUser(profile);
          localStorage.setItem('automl_user', JSON.stringify(profile));
        } catch {
          // Token expired or invalid
          localStorage.removeItem('automl_token');
          localStorage.removeItem('automl_user');
          setToken(null);
          setUser(null);
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = async (payload: LoginPayload) => {
    setIsLoading(true);
    try {
      const res = await authApi.login(payload);
      setToken(res.access_token);
      setUser(res.user);
      localStorage.setItem('automl_token', res.access_token);
      localStorage.setItem('automl_user', JSON.stringify(res.user));
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (payload: RegisterPayload): Promise<User> => {
    return await authApi.register(payload);
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('automl_token');
    localStorage.removeItem('automl_user');
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token && !!user,
        isLoading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
