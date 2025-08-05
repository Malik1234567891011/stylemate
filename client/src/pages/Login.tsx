import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Closet.css"; // reuse basic styles for now

const Login: React.FC = () => {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();

    // 🔐 Temporary validation logic (mocking real login)
    if (email === "test@example.com" && password === "password") {
      // You’d replace this with a real auth flow later
      localStorage.setItem("user", email);
      navigate("/closet");
    } else {
      setError("Invalid email or password.");
    }
  };

  return (
    <div className="closet-container fade-in">
      <div className="closet-card-container">
        <h2 className="closet-title">👤 Login</h2>
        <form onSubmit={handleLogin}>
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
          <button
            type="submit"
            className="upload-btn"
            style={{ width: "100%" }}
          >
            Login
          </button>
        </form>
        {error && (
          <p style={{ color: "crimson", marginTop: "1rem", textAlign: "center" }}>
            {error}
          </p>
        )}
        <p style={{ textAlign: "center", marginTop: "1rem" }}>
          Don’t have an account?{" "}
          <span
            onClick={() => navigate("/register")}
            style={{ color: "#00ccff", cursor: "pointer", textDecoration: "underline" }}
          >
            Register
          </span>
        </p>
      </div>
    </div>
  );
};

export default Login;
