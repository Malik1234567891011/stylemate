import { useState } from "react";
import axios from "axios";
import bonsaiVideo from "../assets/bonsai.mp4";

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
  const [hasRated, setHasRated] = useState(false);
  const [page, setPage] = useState(1);
  const [lastUploadPage, setLastUploadPage] = useState(0);
  const [showTips, setShowTips] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResults(null);
      setError(null);
      setHasRated(false);
      setPage(1);
      setLastUploadPage(0);
    }
  };

  const handleUpload = async (pageOverride?: number) => {
    if (!file) {
      setError("Please select an image.");
      return;
    }

    const currentPage = pageOverride || page;

    if (currentPage === lastUploadPage) {
      setError("You're already viewing this set of recommendations.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post<Product[]>(
        `http://127.0.0.1:8000/recommend?k=5&page=${currentPage}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResults(res.data);
      setPage(currentPage + 1);
      setLastUploadPage(currentPage);
      setHasRated(false);
    } catch (err) {
      console.error(err);
      setError("Error fetching recommendations.");
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (liked: boolean) => {
    if (!file || hasRated) return;

    setHasRated(true);
    try {
      await axios.post("http://127.0.0.1:8000/feedback", {
        filename: file.name,
        feedback: liked ? "like" : "dislike",
        page: lastUploadPage,
      });
    } catch (err) {
      console.error("Feedback submission failed", err);
    }
  };

  const getSimilarityLabel = (score: number): string => {
    if (score > 0.75) return "Identical / Same Item";
    if (score > 0.6) return "Very Similar";
    if (score > 0.4) return "Similar Vibe";
    return "Different Style";
  };

  return (
    <div className="upload-page-wrapper">
      <video
        autoPlay
        muted
        loop
        playsInline
        src={bonsaiVideo}
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          objectFit: "cover",
          zIndex: -1,
          filter: "brightness(0.6)",
        }}
      />

      <div className="upload-container">
        <h2 style={{ color: "#FFF" }}>
          Upload an Outfit Photo{" "}
          <span
            onClick={() => setShowTips(!showTips)}
            style={{
              marginLeft: "0.5rem",
              cursor: "pointer",
              color: "#CCC",
              fontSize: "1.2rem",
            }}
            title="Click for tips"
          >
            ❓
          </span>
        </h2>

        {showTips && (
          <div
            style={{
              background: "#333",
              color: "#EEE",
              padding: "1rem",
              borderRadius: "8px",
              marginTop: "1rem",
              fontSize: "0.9rem",
            }}
          >
            <strong>Upload Tips:</strong>
            <ul>
              <li>Use a clean, non-busy background</li>
              <li>Isolate one outfit per photo</li>
              <li>Avoid blurry or low-light images</li>
              <li>Try to stand upright, not seated/crouched</li>
            </ul>
          </div>
        )}

        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ marginTop: "1rem" }}
        />

        {file && (
          <button
            onClick={() => handleUpload(1)}
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
          <>
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
                      Match: {getSimilarityLabel(prod.score)}
                    </p>
                  </a>
                </div>
              ))}
            </div>

            {!hasRated && (
              <div style={{ marginTop: "2rem", textAlign: "center" }}>
                <p style={{ color: "#FFF", marginBottom: "0.5rem" }}>
                  Did you like these recommendations?
                </p>
                <button
                  onClick={() => sendFeedback(true)}
                  style={{ marginRight: "1rem" }}
                >
                  👍 Yes
                </button>
                <button onClick={() => sendFeedback(false)}>👎 No</button>
              </div>
            )}

            <div style={{ marginTop: "1rem", textAlign: "center" }}>
              <button onClick={() => handleUpload()} disabled={loading}>
                🔁 Get Next 5 Recommendations
              </button>
            </div>
          </>
        )}
      </div>

      <style jsx>{`
        .upload-page-wrapper {
          min-height: 100vh;
          width: 100%;
          background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .upload-container {
          padding: 2rem;
          max-width: 600px;
          width: 100%;
          border-radius: 12px;
          background-color: rgba(0, 0, 0, 0.6);
          color: #fff;
        }

        .upload-container input,
        .upload-container button {
          font-size: 1rem;
        }
      `}</style>
    </div>
  );
};

export default StyleUpload;
