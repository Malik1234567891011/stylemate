// client/src/components/StyleUpload.tsx

import { useState } from "react";
import axios from "axios";
import bonsaiVideo from "../assets/bonsai.mp4"; // <- Make sure this path is correct

interface Product {
  title: string;
  price: string | null;
  url: string;
  score: number;
}

const StyleUpload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<Product[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResults(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post<Product[]>(
        "http://127.0.0.1:8000/recommend?k=5",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResults(res.data);
    } catch (err: any) {
      console.error(err);
      setError("Error fetching recommendations.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Background video */}
      <video
        className="bg-video"
        src={bonsaiVideo}
        autoPlay
        loop
        muted
        playsInline
      />

      {/* Overlay container for the upload UI */}
      <div className="upload-container">
        <h2 style={{ color: "#FFF" }}>Upload an Outfit Photo</h2>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{
            marginTop: "1rem",
          }}
        />

        {file && (
          <button
            onClick={handleUpload}
            disabled={loading}
            style={{
              marginTop: "1rem",
              padding: "0.5rem 1rem",
              background: "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Analyzing..." : "Get Recommendations"}
          </button>
        )}

        {error && (
          <div style={{ marginTop: "1rem", color: "crimson" }}>{error}</div>
        )}

        {results && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
              gap: "1rem",
              marginTop: "2rem",
            }}
          >
            {results.map((prod) => (
              <div
                key={prod.url}
                style={{
                  border: "1px solid #444",
                  padding: "1rem",
                  borderRadius: "8px",
                  background: "#1E1E1E",
                  color: "#E0E0E0",
                }}
              >
                <a
                  href={prod.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ textDecoration: "none", color: "inherit" }}
                >
                  <h3 style={{ marginBottom: "0.5rem", fontSize: "1.1rem" }}>
                    {prod.title}
                  </h3>
                  <p style={{ margin: "0.25rem 0" }}>
                    {prod.price ?? "Price not available"}
                  </p>
                  <p
                    style={{
                      margin: "0.25rem 0",
                      fontSize: "0.9rem",
                      opacity: 0.8,
                    }}
                  >
                    Score: {prod.score.toFixed(3)}
                  </p>
                </a>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Inline CSS */}
      <style jsx>{`
        /* The video is fixed behind everything and covers the viewport */
        .bg-video {
          position: fixed;
          top: 0;
          left: 0;
          width: 100vw;
          height: 100vh;
          object-fit: cover;
          z-index: -1;
          filter: brightness(0.6);
        }

        /* Center and stack the upload UI on top */
        .upload-container {
          position: relative;
          z-index: 1;
          padding: 2rem;
          max-width: 600px;
          margin: 0 auto;
          color: #fff;
        }

        /* Make sure inputs and buttons are visible on top of the video */
        .upload-container input,
        .upload-container button {
          font-size: 1rem;
        }
      `}</style>
    </>
  );
};

export default StyleUpload;
