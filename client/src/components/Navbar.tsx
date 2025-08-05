// src/components/Navbar.tsx
import React from "react";
import { Link } from "react-router-dom";
import "../styles/Navbar.css"; // You'll style this later

const Navbar: React.FC = () => {
  return (
    <nav className="navbar">
      <Link to="/" className="nav-logo">StyleMate</Link>
      <div className="nav-links">
        <Link to="/closet">Closet</Link>
        <Link to="/recommendations">Recommendations</Link>
        <Link to="/login">Login</Link>
        <Link to="/register">Register</Link>
      </div>
    </nav>
  );
};

export default Navbar;
