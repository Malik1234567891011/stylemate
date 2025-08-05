import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Closet.css"; // reusing your styles

const Register: React.FC = () => {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    // 👤 Store dummy account in localStorage (fake backend logic)
    localStorage.setItem("user", email);
    navigate("/closet");
  };

  return (
    <div className="closet-container fade-in">
      <div className="closet-card-container">
        <h2 className="closet-title">🆕 Register</h2>
        <form onSubmit={handleRegister}>
          <div style={{ marginBottom: "1rem" }}>
            <label>Email:</label>
            <input
              type="email"
              value={email}
              required
              onChange={(e) => setEmail(e.target.value)}
              style={{ width: "100%", padding: "0.5rem" }}
            />
          </div>
          <div style={{ marginBottom: "1rem" }}>
            <label>Password:</label>
            <input
              type="password"
              value={password}
              required
              onChange={(e) => setPassword(e.target.value)}
              style={{ width: "100%", padding: "0.5rem" }}
            />
          </div>
          <div style={{ marginBottom: "1rem" }}>
            <label>Confirm Password:</label>
            <input
              type="password"
              value={confirmPassword}
              required
              onChange={(e) => setConfirmPassword(e.target.value)}
              style={{ width: "100%", padding: "0.5rem" }}
            />
          </div>
          <button
            type="submit"
            className="upload-btn"
            style={{ width: "100%" }}
          >
            Create Account
          </button>
        </form>
        {error && (
          <p style={{ color: "crimson", marginTop: "1rem", textAlign: "center" }}>
            {error}
          </p>
        )}
        <p style={{ textAlign: "center", marginTop: "1rem" }}>
          Already have an account?{" "}
          <span
            onClick={() => navigate("/login")}
            style={{ color: "#00ccff", cursor: "pointer", textDecoration: "underline" }}
          >
            Login
          </span>
        </p>
      </div>
    </div>
  );
};

export default Register;
