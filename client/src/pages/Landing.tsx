import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Landing.css';

const Landing: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-container">
      <h1 className="landing-title">👗 Welcome to StyleMate</h1>
      <p className="landing-subtitle">Your AI-powered personal stylist.</p>

      <div className="landing-buttons">
        <button onClick={() => navigate('/recommendations')} className="landing-btn">
          Outfit Finder
        </button>
        <button onClick={() => navigate('/closet')} className="landing-btn secondary">
          View Closet / Expand Wardrobe
        </button>
      </div>
    </div>
  );
};

export default Landing;
