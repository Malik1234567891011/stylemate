import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../styles/Closet.css';

interface ClosetItem {
  vector: number[];
  timestamp: number;
  filename: string;
  type?: string;
}

interface SuggestionItem {
  image_url: string;
  title: string;
  url: string;
  score: number;
}

const Closet: React.FC = () => {
  const [closet, setCloset] = useState<ClosetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<SuggestionItem[]>([]);
  const [suggestionError, setSuggestionError] = useState<string | null>(null);

  // 🔁 Fetch closet
  const fetchCloset = () => {
    fetch('http://127.0.0.1:8000/closet')
      .then((res) => res.json())
      .then((data) => {
        setCloset(data.reverse());
        setLoading(false);
      })
      .catch((err) => {
        console.error('❌ Failed to fetch closet:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    setMounted(true);
    fetchCloset();
  }, []);

  // ⬆️ Handle image upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("saveImage", "true");

    axios
      .post("http://127.0.0.1:8000/upload_fit", formData)
      .then(() => {
        setUploading(false);
        fetchCloset();
      })
      .catch((err) => {
        console.error("❌ Upload error:", err);
        setUploadError("Upload failed. Try again.");
        setUploading(false);
      });
  };

  // 🤖 Fetch AI suggestions
  const fetchSuggestions = () => {
    setSuggestionError(null);
    fetch("http://127.0.0.1:8000/recommend_expand?k=5")
      .then((res) => {
        if (!res.ok) throw new Error("Suggestion fetch failed");
        return res.json();
      })
      .then((data) => setSuggestions(data))
      .catch((err) => {
        console.error("❌ Suggestion fetch error:", err);
        setSuggestionError("Could not fetch suggestions.");
      });
  };

  // ❌ Delete outfit from closet
  const handleDelete = (index: number) => {
    const updated = [...closet];
    updated.splice(index, 1);
    setCloset(updated);

    fetch("http://127.0.0.1:8000/closet", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updated),
    }).catch((err) => {
      console.error("❌ Delete failed:", err);
    });
  };

  return (
    <div className={`closet-container ${mounted ? 'fade-in' : ''}`}>
      <div className="closet-card-container">
        <h2 className="closet-title">🧳 Your Closet</h2>

        {/* 📤 Upload New Fit */}
        <label className="upload-btn" style={{ marginBottom: "1rem" }}>
          Upload New Fit
          <input
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
        </label>
        {uploading && <p className="loading-text">Uploading...</p>}
        {uploadError && <p className="error-text">{uploadError}</p>}

        {/* 🤖 Suggestion Button */}
        {closet.length > 0 && (
          <button className="upload-btn" style={{ marginBottom: "2rem" }} onClick={fetchSuggestions}>
            Get AI Suggestions
          </button>
        )}

        {/* 👕 Closet Grid */}
        {loading ? (
          <p className="loading-text">Loading your saved outfits...</p>
        ) : closet.length === 0 ? (
          <div className="empty-closet-card">
            <p className="empty-text">Your closet’s looking empty right now.</p>
          </div>
        ) : (
          <div className="closet-grid">
            {closet.map((item, i) => (
              <div key={i} className="closet-card">
                <div className="closet-img-placeholder">
                  {item.filename ? (
                    <img
                      src={`http://127.0.0.1:8000/uploads/${item.filename}`}
                      alt={`Outfit ${i + 1}`}
                      className="closet-img"
                    />
                  ) : (
                    <span>🖼️</span>
                  )}
                </div>
                <div className="closet-info">
                  <p className="closet-filename">Uploaded Outfit #{i + 1}</p>
                  <p className="closet-timestamp">
                    {new Date(item.timestamp * 1000).toLocaleString()}
                  </p>
                  <button
                    className="delete-btn"
                    onClick={() => handleDelete(i)}
                  >
                    ❌ Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* 🎯 AI Recommendations */}
        {suggestions.length > 0 && (
          <>
            <h3 className="suggestion-title">🧠 Suggested Items</h3>
            <div className="closet-grid">
              {suggestions.map((item, i) => (
                <a
                  key={i}
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="closet-card"
                >
                  <div className="closet-img-placeholder">
                    <img src={item.image_url} alt={item.title} className="closet-img" />
                  </div>
                  <div className="closet-info">
                    <p className="closet-filename">{item.title}</p>
                  </div>
                </a>
              ))}
            </div>
          </>
        )}
        {suggestionError && (
          <p className="error-text">{suggestionError}</p>
        )}
      </div>
    </div>
  );
};

export default Closet;
